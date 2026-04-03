from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from .openvla_protocol import build_openvla_prompt, decode_image_from_base64, to_jsonable_action

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _resolve_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    dtype = mapping[dtype_name]
    if device.type == "cpu" and dtype is not torch.float32:
        logger.warning("Downgrading OpenVLA runtime dtype from %s to float32 on CPU.", dtype_name)
        return torch.float32
    return dtype


def _resolve_attn_implementation(attn_implementation: str | None, device: torch.device) -> str | None:
    if attn_implementation != "flash_attention_2":
        return attn_implementation

    if device.type != "cuda":
        logger.warning("Disabling flash_attention_2 because OpenVLA runtime is running on %s.", device.type)
        return None

    return attn_implementation


def _resolve_runtime_quantization(
    runtime_quantization: str,
    device: torch.device,
) -> str:
    normalized = runtime_quantization.strip().lower()
    if normalized not in {"none", "8bit", "4bit"}:
        raise ValueError(
            "Unsupported runtime quantization: "
            f"{runtime_quantization}. Expected one of: none, 8bit, 4bit."
        )
    if normalized != "none" and device.type != "cuda":
        logger.warning(
            "Disabling %s quantization because the OpenVLA runtime is running on %s.",
            normalized,
            device.type,
        )
        return "none"
    return normalized


def _build_quantization_kwargs(
    runtime_quantization: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> dict[str, Any]:
    if runtime_quantization == "none":
        return {}

    transformers = importlib.import_module("transformers")
    bits_and_bytes_config = getattr(transformers, "BitsAndBytesConfig", None)
    if bits_and_bytes_config is None:
        raise RuntimeError(
            "runtime_quantization requires transformers.BitsAndBytesConfig. "
            "Install a runtime with bitsandbytes support."
        )

    quantization_kwargs: dict[str, Any] = {}
    if runtime_quantization == "8bit":
        quantization_kwargs["load_in_8bit"] = True
    else:
        quantization_kwargs.update(
            {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch_dtype,
                "bnb_4bit_quant_type": "nf4",
            }
        )

    device_index = device.index if device.index is not None else 0
    return {
        "quantization_config": bits_and_bytes_config(**quantization_kwargs),
        "device_map": {"": device_index},
    }


def _enable_quantized_force_hooks() -> None:
    """Force accelerate hook dispatch for quantized models on a single device.

    Transformers 4.40.x routes quantized single-device loads through
    ``dispatch_model(...)->model.to(device)`` unless ``force_hooks`` is set.
    bitsandbytes < 0.43.2 rejects that ``.to(...)`` call for 4-bit models, so
    patch the local dispatch entrypoint used by ``from_pretrained`` to keep the
    quantized path on the hook-based branch.
    """

    modeling_utils = importlib.import_module("transformers.modeling_utils")
    if getattr(modeling_utils, "_roboneuron_quantized_force_hooks_enabled", False):
        return

    original_dispatch_model = getattr(modeling_utils, "dispatch_model", None)
    if original_dispatch_model is None:
        raise RuntimeError("transformers.modeling_utils.dispatch_model is unavailable.")

    def _dispatch_model_with_force_hooks(model, *args, **kwargs):
        kwargs.setdefault("force_hooks", True)
        return original_dispatch_model(model, *args, **kwargs)

    modeling_utils.dispatch_model = _dispatch_model_with_force_hooks
    modeling_utils._roboneuron_quantized_force_hooks_enabled = True


class OpenVLAWorker:
    def __init__(
        self,
        *,
        model_path: str,
        attn_implementation: str | None,
        dtype_name: str,
        device_name: str,
        runtime_quantization: str,
        low_cpu_mem_usage: bool,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve(strict=False))
        self.low_cpu_mem_usage = low_cpu_mem_usage

        if device_name == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
        self.torch_dtype = _resolve_dtype(dtype_name, self.device)
        self.attn_implementation = _resolve_attn_implementation(attn_implementation, self.device)
        self.runtime_quantization = _resolve_runtime_quantization(runtime_quantization, self.device)

        self.processor = None
        self.model = None

    def load(self) -> None:
        logger.info("Loading OpenVLA runtime from %s on %s", self.model_path, self.device)

        self.processor = self._load_processor()
        if self.runtime_quantization != "none":
            _enable_quantized_force_hooks()
        self.model = OpenVLAForActionPrediction.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.low_cpu_mem_usage,
            local_files_only=True,
            **_build_quantization_kwargs(
                self.runtime_quantization,
                device=self.device,
                torch_dtype=self.torch_dtype,
            ),
        )
        if self.runtime_quantization == "none":
            self.model = self.model.to(self.device, dtype=self.torch_dtype)

        stats_path = Path(self.model_path) / "dataset_statistics.json"
        if stats_path.is_file():
            with stats_path.open("r", encoding="utf-8") as handle:
                self.model.norm_stats = json.load(handle)

    def _load_processor(self) -> PrismaticProcessor:
        from transformers import AutoTokenizer

        image_processor = PrismaticImageProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        return PrismaticProcessor(image_processor=image_processor, tokenizer=tokenizer)

    def predict_action(
        self,
        *,
        image_base64: str,
        instruction: str,
        unnorm_key: str | None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if self.model is None or self.processor is None:
            raise RuntimeError("OpenVLA runtime is not loaded.")

        prompt = build_openvla_prompt(instruction, self.model_path)
        image = decode_image_from_base64(image_base64)
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch_dtype)
        predict_kwargs = dict(kwargs or {})
        action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False, **predict_kwargs)
        return to_jsonable_action(action)


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="OpenVLA subprocess worker")
    parser.add_argument("--model-path", required=True, help="Path to the local OpenVLA checkpoint")
    parser.add_argument("--attn-implementation", default=None, help="Transformers attention backend")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", help="Runtime device spec, e.g. auto / cuda:0 / cpu")
    parser.add_argument("--runtime-quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Enable transformers low_cpu_mem_usage")
    args = parser.parse_args()

    worker = OpenVLAWorker(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        dtype_name=args.dtype,
        device_name=args.device,
        runtime_quantization=args.runtime_quantization,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
    )

    try:
        worker.load()
        norm_keys = sorted(getattr(worker.model, "norm_stats", {}).keys()) if worker.model is not None else []
        _emit(
            {
                "event": "ready",
                "device": str(worker.device),
                "dtype": str(worker.torch_dtype),
                "runtime_quantization": worker.runtime_quantization,
                "norm_keys": norm_keys,
            }
        )
    except Exception as exc:
        logger.error("Failed to load OpenVLA worker: %s", exc)
        _emit(
            {
                "event": "startup_error",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )
        raise

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue

        request: dict[str, Any]
        try:
            request = json.loads(line)
            request_id = request["id"]
            method = request["method"]
            params = request.get("params", {})

            if method == "predict_action":
                action = worker.predict_action(**params)
                _emit({"id": request_id, "ok": True, "result": {"action": action}})
            elif method == "shutdown":
                _emit({"id": request_id, "ok": True, "result": {"status": "bye"}})
                break
            else:
                raise ValueError(f"Unsupported method: {method}")
        except Exception as exc:
            _emit(
                {
                    "id": request.get("id"),
                    "ok": False,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                }
            )


if __name__ == "__main__":
    main()
