from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import logging
import sys
import traceback
import types
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .openvla_oft_protocol import decode_observation_from_transport
from .openvla_protocol import build_openvla_prompt, to_jsonable_action

logger = logging.getLogger(__name__)

_PROFILE_TO_NUM_IMAGES = {
    "bridge": 1,
    "libero": 2,
    "aloha": 3,
}
_PROFILE_TO_PROPRIO_DIM = {
    "bridge": 7,
    "libero": 8,
    "aloha": 14,
}
_PROFILE_TO_NORM_KIND = {
    "bridge": "bounds_q99",
    "libero": "bounds_q99",
    "aloha": "bounds",
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def _ensure_shallow_package(name: str, package_path: Path) -> None:
    if name in sys.modules:
        return

    module = types.ModuleType(name)
    module.__file__ = str(package_path / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(package_path)]  # type: ignore[attr-defined]
    spec = importlib.machinery.ModuleSpec(name=name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(package_path)]
    module.__spec__ = spec
    sys.modules[name] = module


def _prepare_prismatic_shallow_packages() -> None:
    prismatic_root = _project_root() / "third_party" / "vla_src" / "openvla-oft" / "prismatic"
    package_map = {
        "prismatic": prismatic_root,
        "prismatic.models": prismatic_root / "models",
        "prismatic.vla": prismatic_root / "vla",
        "prismatic.training": prismatic_root / "training",
        "prismatic.overwatch": prismatic_root / "overwatch",
        "prismatic.extern": prismatic_root / "extern",
        "prismatic.extern.hf": prismatic_root / "extern" / "hf",
    }
    for package_name, package_path in package_map.items():
        _ensure_shallow_package(package_name, package_path)


def _str_to_optional_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value!r}")


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
        logger.warning("Downgrading OpenVLA-OFT runtime dtype from %s to float32 on CPU.", dtype_name)
        return torch.float32
    return dtype


def _resolve_attn_implementation(attn_implementation: str | None, device: torch.device) -> str | None:
    if attn_implementation != "flash_attention_2":
        return attn_implementation
    if device.type != "cuda":
        logger.warning(
            "Disabling flash_attention_2 because OpenVLA-OFT runtime is running on %s.", device.type
        )
        return None

    try:
        from transformers.utils import is_flash_attn_2_available
    except Exception:
        is_available = False
    else:
        is_available = bool(is_flash_attn_2_available())

    if not is_available:
        logger.warning("Disabling flash_attention_2 because Flash Attention 2 is not available in this runtime.")
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
            "Disabling %s quantization because the OpenVLA-OFT runtime is running on %s.",
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


def _load_json_if_present(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _proprio_dim_from_stats(stats: dict[str, Any] | None) -> int | None:
    if not stats:
        return None

    for entry in stats.values():
        if not isinstance(entry, dict):
            continue
        proprio = entry.get("proprio")
        if isinstance(proprio, dict):
            for key in ("q99", "max", "mean", "min"):
                values = proprio.get(key)
                if isinstance(values, list) and values:
                    return len(values)
    return None


def _infer_robot_platform(model_path: Path, requested: str | None) -> str:
    if requested:
        resolved = requested.strip().lower()
        if resolved not in _PROFILE_TO_NUM_IMAGES:
            raise ValueError(f"Unsupported OpenVLA-OFT robot platform: {requested}")
        return resolved

    model_path_str = str(model_path).lower()
    for profile in _PROFILE_TO_NUM_IMAGES:
        if profile in model_path_str:
            return profile

    dataset_stats = _load_json_if_present(model_path / "dataset_statistics.json")
    proprio_dim = _proprio_dim_from_stats(dataset_stats)
    if proprio_dim is None:
        config_json = _load_json_if_present(model_path / "config.json")
        if config_json is not None:
            proprio_dim = _proprio_dim_from_stats(config_json.get("norm_stats"))

    reverse_map = {value: key for key, value in _PROFILE_TO_PROPRIO_DIM.items()}
    return reverse_map.get(proprio_dim, "bridge")


def _activate_robot_platform(profile: str) -> None:
    argv_text = " ".join(sys.argv).lower()
    if profile not in argv_text:
        sys.argv.append(profile)


def _sorted_checkpoint_candidates(model_path: Path, prefix: str) -> list[Path]:
    candidates = sorted(model_path.glob(f"{prefix}--*_checkpoint.pt"))
    if not candidates:
        return []

    def key(path: Path) -> tuple[int, str]:
        stem = path.stem
        middle = stem.split("--", 1)[1].rsplit("_checkpoint", 1)[0]
        if middle == "latest":
            return (10**12, middle)
        digits = "".join(ch for ch in middle if ch.isdigit())
        return (int(digits) if digits else -1, middle)

    return sorted(candidates, key=key)


def _load_component_state_dict(path: Path) -> dict[str, Any]:
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }


def _normalize_proprio(proprio: np.ndarray, stats: dict[str, Any], profile: str) -> np.ndarray:
    norm_kind = _PROFILE_TO_NORM_KIND[profile]
    if norm_kind == "bounds":
        high = np.asarray(stats["max"], dtype=np.float32)
        low = np.asarray(stats["min"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", np.ones_like(high, dtype=bool)), dtype=bool)
    elif norm_kind == "bounds_q99":
        high = np.asarray(stats["q99"], dtype=np.float32)
        low = np.asarray(stats["q01"], dtype=np.float32)
        mask = np.asarray(stats.get("mask", np.ones_like(high, dtype=bool)), dtype=bool)
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported proprio normalization kind: {norm_kind}")

    normalized = np.where(
        mask,
        2 * (proprio - low) / (high - low + 1e-8) - 1,
        proprio,
    )
    return np.clip(normalized, a_min=-1.0, a_max=1.0).astype(np.float32)


def _apply_center_crop(image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * crop_scale)))
    crop_height = max(1, int(round(height * crop_scale)))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image.crop((left, top, left + crop_width, top + crop_height))


def _prepare_image(image: Image.Image, size: int, center_crop: bool) -> Image.Image:
    prepared = image.convert("RGB")
    if prepared.size != (size, size):
        prepared = prepared.resize((size, size), resample=Image.Resampling.LANCZOS)
    if center_crop:
        prepared = _apply_center_crop(prepared).resize((size, size), resample=Image.Resampling.LANCZOS)
    return prepared


def _emit(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


class OpenVLAOFTWorker:
    def __init__(
        self,
        *,
        model_path: str,
        attn_implementation: str | None,
        dtype_name: str,
        device_name: str,
        runtime_quantization: str,
        low_cpu_mem_usage: bool,
        use_l1_regression: bool | None,
        use_diffusion: bool | None,
        use_film: bool | None,
        use_proprio: bool | None,
        num_images_in_input: int | None,
        num_diffusion_steps_inference: int,
        lora_rank: int,
        center_crop: bool,
        unnorm_key: str | None,
        robot_platform: str | None,
        default_proprio: list[float] | None,
        base_model_path: str | None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve(strict=False)
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.dataset_stats = _load_json_if_present(self.model_path / "dataset_statistics.json")
        self.robot_platform = _infer_robot_platform(self.model_path, robot_platform)
        _activate_robot_platform(self.robot_platform)

        if device_name == "auto":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)
        self.torch_dtype = _resolve_dtype(dtype_name, self.device)
        self.attn_implementation = _resolve_attn_implementation(attn_implementation, self.device)
        self.runtime_quantization = _resolve_runtime_quantization(runtime_quantization, self.device)
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_film = use_film
        self.use_proprio = use_proprio
        self.num_images_in_input = num_images_in_input
        self.num_diffusion_steps_inference = num_diffusion_steps_inference
        self.lora_rank = lora_rank
        self.center_crop = center_crop
        self.default_unnorm_key = unnorm_key
        self.default_proprio = np.asarray(default_proprio, dtype=np.float32) if default_proprio else None
        self.base_model_path = Path(base_model_path).expanduser().resolve(strict=False) if base_model_path else None

        self.processor = None
        self.model = None
        self.action_head = None
        self.noisy_action_projector = None
        self.proprio_projector = None
        self._runtime_modules: dict[str, Any] | None = None
        self.image_size = 224
        self.resolved_num_images_in_input = num_images_in_input or _PROFILE_TO_NUM_IMAGES[self.robot_platform]
        self.resolved_use_l1_regression = False
        self.resolved_use_diffusion = False
        self.resolved_use_film = False
        self.resolved_use_proprio = False

    def load(self) -> None:
        with redirect_stdout(sys.stderr):
            modules = self._runtime_modules or self._import_runtime_modules()
            self._runtime_modules = modules
            self.resolved_num_images_in_input = self.num_images_in_input or _PROFILE_TO_NUM_IMAGES[self.robot_platform]
            self.resolved_use_diffusion = (
                self.use_diffusion
                if self.use_diffusion is not None
                else bool(_sorted_checkpoint_candidates(self.model_path, "noisy_action_projector"))
            )
            self.resolved_use_l1_regression = (
                self.use_l1_regression
                if self.use_l1_regression is not None
                else bool(_sorted_checkpoint_candidates(self.model_path, "action_head")) and not self.resolved_use_diffusion
            )
            self.resolved_use_film = (
                self.use_film
                if self.use_film is not None
                else bool(_sorted_checkpoint_candidates(self.model_path, "vision_backbone"))
            )
            self.resolved_use_proprio = (
                self.use_proprio
                if self.use_proprio is not None
                else bool(_sorted_checkpoint_candidates(self.model_path, "proprio_projector"))
            )

            logger.info("Loading OpenVLA-OFT runtime from %s on %s", self.model_path, self.device)

            self.processor = self._load_processor(modules)
            self.model = self._load_vla_model()
            self.model.vision_backbone.set_num_images_in_input(self.resolved_num_images_in_input)

            if self.resolved_use_film:
                checkpoint = self._latest_component_checkpoint("vision_backbone")
                vision_backbone_state_dict = _load_component_state_dict(checkpoint)
                if self._state_dict_uses_lora(vision_backbone_state_dict):
                    self.model = self._apply_lora_scaffold(self.model)
                backbone_host = self._vision_backbone_host(self.model)
                backbone_host.vision_backbone = modules["FiLMedPrismaticVisionBackbone"](
                    vision_backbone=backbone_host.vision_backbone,
                    llm_dim=self.model.llm_dim,
                )
                backbone_host.vision_backbone.load_state_dict(vision_backbone_state_dict)

            self.model.eval()
            if self.runtime_quantization == "none":
                self.model = self.model.to(self.device, dtype=self.torch_dtype)
            self.model.vision_backbone.set_num_images_in_input(self.resolved_num_images_in_input)

            if self.dataset_stats:
                self._set_model_norm_stats(self.dataset_stats)

            norm_stats = self._get_model_norm_stats()
            if self.default_unnorm_key is None and isinstance(norm_stats, dict) and len(norm_stats) == 1:
                self.default_unnorm_key = next(iter(norm_stats.keys()))

            if hasattr(self.processor, "image_processor"):
                input_sizes = getattr(self.processor.image_processor, "input_sizes", None)
                if isinstance(input_sizes, list) and input_sizes:
                    image_size = input_sizes[0][-1]
                    if isinstance(image_size, int):
                        self.image_size = image_size

            if self.resolved_use_proprio:
                checkpoint = self._latest_component_checkpoint("proprio_projector")
                proprio_stats = self._resolve_proprio_stats(self.default_unnorm_key)
                proprio_dim = _proprio_dim_from_stats(
                    {"resolved": {"proprio": proprio_stats}}
                ) or _PROFILE_TO_PROPRIO_DIM[self.robot_platform]
                self.proprio_projector = modules["ProprioProjector"](
                    llm_dim=self.model.llm_dim,
                    proprio_dim=proprio_dim,
                )
                self.proprio_projector.load_state_dict(_load_component_state_dict(checkpoint))
                self.proprio_projector.eval()
                self.proprio_projector = self.proprio_projector.to(self.device, dtype=self.torch_dtype)

            if self.resolved_use_diffusion:
                checkpoint = self._latest_component_checkpoint("noisy_action_projector")
                self.noisy_action_projector = modules["NoisyActionProjector"](llm_dim=self.model.llm_dim)
                self.noisy_action_projector.load_state_dict(_load_component_state_dict(checkpoint))
                self.noisy_action_projector.eval()
                self.noisy_action_projector = self.noisy_action_projector.to(self.device, dtype=self.torch_dtype)

            if self.resolved_use_l1_regression or self.resolved_use_diffusion:
                checkpoint = self._latest_component_checkpoint("action_head")
                if self.resolved_use_diffusion:
                    self.action_head = modules["DiffusionActionHead"](
                        input_dim=self.model.llm_dim,
                        hidden_dim=self.model.llm_dim,
                        num_diffusion_steps_train=self.num_diffusion_steps_inference,
                    )
                    self.action_head.noise_scheduler.set_timesteps(self.num_diffusion_steps_inference)
                else:
                    self.action_head = modules["L1RegressionActionHead"](
                        input_dim=self.model.llm_dim,
                        hidden_dim=self.model.llm_dim,
                    )
                self.action_head.load_state_dict(_load_component_state_dict(checkpoint))
                self.action_head.eval()
                self.action_head = self.action_head.to(self.device, dtype=self.torch_dtype)

    def predict_action(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        unnorm_key: str | None,
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if self.model is None or self.processor is None:
            raise RuntimeError("OpenVLA-OFT runtime is not loaded.")

        decoded_observation = decode_observation_from_transport(observation)
        request_kwargs = dict(kwargs or {})
        prompt_text = instruction or str(decoded_observation.get("instruction", ""))
        final_unnorm_key = unnorm_key or self.default_unnorm_key
        if final_unnorm_key is None:
            raise ValueError("OpenVLA-OFT requires an unnorm_key for action denormalization.")

        images = self._collect_images(decoded_observation, request_kwargs)
        primary_image = _prepare_image(images[0], self.image_size, self.center_crop)
        inputs = self.processor(build_openvla_prompt(prompt_text, self.model_path), primary_image).to(
            self.device,
            dtype=self.torch_dtype,
        )

        extra_images = images[1:]
        if extra_images:
            wrist_inputs = [
                self.processor(
                    build_openvla_prompt(prompt_text, self.model_path),
                    _prepare_image(image, self.image_size, self.center_crop),
                ).to(self.device, dtype=self.torch_dtype)
                for image in extra_images
            ]
            stacked_pixel_values = [inputs["pixel_values"]]
            stacked_pixel_values.extend(wrist_input["pixel_values"] for wrist_input in wrist_inputs)
            inputs["pixel_values"] = torch.cat(stacked_pixel_values, dim=1)

        proprio = self._resolve_proprio(decoded_observation, request_kwargs, final_unnorm_key)
        predict_kwargs: dict[str, Any] = {
            "unnorm_key": final_unnorm_key,
            "do_sample": False,
        }
        if self.action_head is not None:
            predict_kwargs["action_head"] = self.action_head
            predict_kwargs["use_film"] = self.resolved_use_film
        if self.proprio_projector is not None and proprio is not None:
            predict_kwargs["proprio_projector"] = self.proprio_projector
            predict_kwargs["proprio"] = proprio
        if self.noisy_action_projector is not None:
            predict_kwargs["noisy_action_projector"] = self.noisy_action_projector

        action = self.model.predict_action(**inputs, **predict_kwargs)
        if isinstance(action, tuple):
            action = action[0]
        return to_jsonable_action(action)

    def _import_runtime_modules(self) -> dict[str, Any]:
        _prepare_prismatic_shallow_packages()
        action_heads = importlib.import_module("prismatic.models.action_heads")
        film_module = importlib.import_module("prismatic.models.film_vit_wrapper")
        projectors = importlib.import_module("prismatic.models.projectors")
        processing_module = importlib.import_module("prismatic.extern.hf.processing_prismatic")
        modeling_module = importlib.import_module("prismatic.extern.hf.modeling_prismatic")
        return {
            "L1RegressionActionHead": action_heads.L1RegressionActionHead,
            "DiffusionActionHead": action_heads.DiffusionActionHead,
            "NoisyActionProjector": projectors.NoisyActionProjector,
            "ProprioProjector": projectors.ProprioProjector,
            "FiLMedPrismaticVisionBackbone": film_module.FiLMedPrismaticVisionBackbone,
            "OpenVLAForActionPrediction": modeling_module.OpenVLAForActionPrediction,
            "PrismaticImageProcessor": processing_module.PrismaticImageProcessor,
            "PrismaticProcessor": processing_module.PrismaticProcessor,
        }

    @staticmethod
    def _state_dict_uses_lora(state_dict: dict[str, Any]) -> bool:
        return any(".lora_" in key or ".base_layer." in key for key in state_dict)

    @staticmethod
    def _vision_backbone_host(model: Any) -> Any:
        return model.model if hasattr(model, "model") else model

    def _apply_lora_scaffold(self, model: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=min(self.lora_rank, 16),
            lora_dropout=0.0,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        return get_peft_model(model, lora_config)

    def _latest_component_checkpoint(self, prefix: str) -> Path:
        candidates = _sorted_checkpoint_candidates(self.model_path, prefix)
        if not candidates:
            raise FileNotFoundError(f"Could not find `{prefix}--*_checkpoint.pt` under {self.model_path}")
        return candidates[-1]

    def _load_processor(self, modules: dict[str, Any]) -> Any:
        tokenizer_class = importlib.import_module("transformers").AutoTokenizer
        image_processor = modules["PrismaticImageProcessor"].from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        tokenizer = tokenizer_class.from_pretrained(
            self.model_path,
            local_files_only=True,
        )
        return modules["PrismaticProcessor"](
            image_processor=image_processor,
            tokenizer=tokenizer,
        )

    def _load_vla_model(self) -> Any:
        modules = self._runtime_modules or self._import_runtime_modules()
        self._runtime_modules = modules
        model_class = modules["OpenVLAForActionPrediction"]
        if self._has_root_model_weights():
            return model_class.from_pretrained(
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

        adapter_dir = self.model_path / "lora_adapter"
        if self.base_model_path is None or not adapter_dir.is_dir():
            raise RuntimeError(
                "OpenVLA-OFT checkpoint is missing merged model weights. "
                "Provide `base_model_path` pointing to the base OpenVLA checkpoint."
            )

        from peft import PeftModel

        base_model = model_class.from_pretrained(
            self.base_model_path,
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
        return PeftModel.from_pretrained(base_model, adapter_dir).merge_and_unload()

    def _has_root_model_weights(self) -> bool:
        direct_candidates = [
            self.model_path / "model.safetensors",
            self.model_path / "model.safetensors.index.json",
            self.model_path / "pytorch_model.bin",
            self.model_path / "pytorch_model.bin.index.json",
        ]
        if any(path.exists() for path in direct_candidates):
            return True
        return any(self.model_path.glob("model-*.safetensors"))

    @staticmethod
    def _iter_wrapped_models(model: Any) -> list[Any]:
        models: list[Any] = []
        queue = [model]
        seen: set[int] = set()
        while queue:
            current = queue.pop(0)
            if current is None or id(current) in seen:
                continue
            seen.add(id(current))
            models.append(current)
            for attr_name in ("model", "base_model"):
                child = getattr(current, attr_name, None)
                if child is not None and child is not current:
                    queue.append(child)
        return models

    def _set_model_norm_stats(self, norm_stats: dict[str, Any]) -> None:
        for current_model in self._iter_wrapped_models(self.model):
            current_model.norm_stats = norm_stats
            config = getattr(current_model, "config", None)
            if config is not None:
                config.norm_stats = norm_stats

    def _get_model_norm_stats(self) -> dict[str, Any] | None:
        for current_model in self._iter_wrapped_models(self.model):
            norm_stats = getattr(current_model, "norm_stats", None)
            if isinstance(norm_stats, dict) and norm_stats:
                return norm_stats
        return None

    def _resolve_proprio_stats(self, unnorm_key: str | None) -> dict[str, Any] | None:
        norm_stats = self._get_model_norm_stats()
        if not isinstance(norm_stats, dict) or not norm_stats:
            return None

        if unnorm_key is None:
            if len(norm_stats) != 1:
                return None
            unnorm_key = next(iter(norm_stats.keys()))

        entry = norm_stats.get(unnorm_key)
        if not isinstance(entry, dict):
            return None
        proprio_stats = entry.get("proprio")
        return proprio_stats if isinstance(proprio_stats, dict) else None

    def _resolve_proprio(
        self,
        observation: dict[str, Any],
        request_kwargs: dict[str, Any],
        unnorm_key: str,
    ) -> np.ndarray | None:
        if not self.resolved_use_proprio:
            return None

        proprio = request_kwargs.pop("proprio", None)
        if proprio is None and "state" in observation:
            proprio = observation["state"]
        if proprio is None and "proprio" in observation:
            proprio = observation["proprio"]

        if proprio is None:
            proprio_stats = self._resolve_proprio_stats(unnorm_key)
            proprio_dim = _proprio_dim_from_stats({"resolved": {"proprio": proprio_stats}}) if proprio_stats else None
            if self.default_proprio is not None:
                proprio = self.default_proprio
            elif proprio_dim is not None:
                logger.warning(
                    "OpenVLA-OFT request missing proprio input; using zeros with dimension %s.", proprio_dim
                )
                proprio = np.zeros((proprio_dim,), dtype=np.float32)
            else:
                raise ValueError("OpenVLA-OFT request missing proprio input and no default proprio is configured.")

        proprio_array = np.asarray(proprio, dtype=np.float32)
        proprio_stats = self._resolve_proprio_stats(unnorm_key)
        if proprio_stats is None:
            return proprio_array
        return _normalize_proprio(proprio_array, proprio_stats, self.robot_platform)

    def _collect_images(self, observation: dict[str, Any], request_kwargs: dict[str, Any]) -> list[Image.Image]:
        explicit_images = request_kwargs.pop("images", None)
        if explicit_images is not None:
            images = list(explicit_images)
        elif "images" in observation and isinstance(observation["images"], list):
            images = list(observation["images"])
        else:
            images: list[Any] = []
            for key in ("full_image", "image", "primary_image"):
                if key in observation:
                    images.append(observation[key])
                    break
            ordered_wrist_keys = [key for key in ("wrist_image", "left_wrist_image", "right_wrist_image") if key in observation]
            remaining_wrist_keys = sorted(
                key
                for key in observation
                if "wrist" in key and key not in ordered_wrist_keys
            )
            for key in ordered_wrist_keys + remaining_wrist_keys:
                images.append(observation[key])

        prepared = [
            image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
            for image in images
        ]
        if not prepared:
            raise ValueError("OpenVLA-OFT request does not contain any images.")

        if len(prepared) < self.resolved_num_images_in_input:
            logger.warning(
                "OpenVLA-OFT expected %s images but received %s; duplicating the last frame.",
                self.resolved_num_images_in_input,
                len(prepared),
            )
            prepared.extend([prepared[-1].copy()] * (self.resolved_num_images_in_input - len(prepared)))
        elif len(prepared) > self.resolved_num_images_in_input:
            prepared = prepared[: self.resolved_num_images_in_input]
        return prepared


def main() -> None:
    _configure_logging()

    parser = argparse.ArgumentParser(description="OpenVLA-OFT subprocess worker")
    parser.add_argument("--model-path", required=True, help="Path to the local OpenVLA-OFT checkpoint")
    parser.add_argument("--attn-implementation", default=None, help="Transformers attention backend")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--device", default="auto", help="Runtime device spec, e.g. auto / cuda:0 / cpu")
    parser.add_argument("--runtime-quantization", default="none", choices=["none", "8bit", "4bit"])
    parser.add_argument("--low-cpu-mem-usage", action="store_true", help="Enable transformers low_cpu_mem_usage")
    parser.add_argument("--use-l1-regression", type=_str_to_optional_bool, default=None)
    parser.add_argument("--use-diffusion", type=_str_to_optional_bool, default=None)
    parser.add_argument("--use-film", type=_str_to_optional_bool, default=None)
    parser.add_argument("--use-proprio", type=_str_to_optional_bool, default=None)
    parser.add_argument("--num-images-in-input", type=int, default=None)
    parser.add_argument("--num-diffusion-steps-inference", type=int, default=50)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--center-crop", dest="center_crop", action="store_true")
    parser.add_argument("--no-center-crop", dest="center_crop", action="store_false")
    parser.set_defaults(center_crop=True)
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--robot-platform", default=None, choices=["bridge", "libero", "aloha"])
    parser.add_argument("--default-proprio-json", default=None)
    parser.add_argument("--base-model-path", default=None)
    args = parser.parse_args()

    worker = OpenVLAOFTWorker(
        model_path=args.model_path,
        attn_implementation=args.attn_implementation,
        dtype_name=args.dtype,
        device_name=args.device,
        runtime_quantization=args.runtime_quantization,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        use_l1_regression=args.use_l1_regression,
        use_diffusion=args.use_diffusion,
        use_film=args.use_film,
        use_proprio=args.use_proprio,
        num_images_in_input=args.num_images_in_input,
        num_diffusion_steps_inference=args.num_diffusion_steps_inference,
        lora_rank=args.lora_rank,
        center_crop=args.center_crop,
        unnorm_key=args.unnorm_key,
        robot_platform=args.robot_platform,
        default_proprio=json.loads(args.default_proprio_json) if args.default_proprio_json else None,
        base_model_path=args.base_model_path,
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
                "robot_platform": worker.robot_platform,
                "num_images_in_input": worker.resolved_num_images_in_input,
                "use_proprio": worker.resolved_use_proprio,
                "use_film": worker.resolved_use_film,
                "use_l1_regression": worker.resolved_use_l1_regression,
                "use_diffusion": worker.resolved_use_diffusion,
            }
        )
    except Exception as exc:
        logger.error("Failed to load OpenVLA-OFT worker: %s", exc)
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
