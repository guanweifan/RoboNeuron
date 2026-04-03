from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from roboneuron_core.runtime.openvla_client import OpenVLASubprocessClient

from .base import ModelWrapper

logger = logging.getLogger(__name__)


class OpenVLAWrapper(ModelWrapper):
    """Thin adapter that proxies OpenVLA inference to a dedicated subprocess runtime."""

    def __init__(
        self,
        model_path: str | Path,
        attn_implementation: str | None = None,
        dtype: torch.dtype | str | None = None,
        runtime_quantization: str = "none",
        default_unnorm_key: str | None = None,
        runtime_python: str | Path | None = None,
        runtime_module: str = "roboneuron_core.runtime.openvla_worker",
        runtime_extra_python_paths: list[str] | tuple[str, ...] | None = None,
        runtime_startup_timeout_sec: float = 900.0,
        runtime_request_timeout_sec: float = 300.0,
        runtime_device: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, **kwargs)
        self.attn_implementation = attn_implementation
        self.default_unnorm_key = default_unnorm_key
        self.torch_dtype = dtype or torch.bfloat16
        self._runtime = OpenVLASubprocessClient(
            model_path=self.model_path,
            runtime_python=runtime_python,
            runtime_module=runtime_module,
            runtime_extra_python_paths=runtime_extra_python_paths,
            startup_timeout_sec=runtime_startup_timeout_sec,
            request_timeout_sec=runtime_request_timeout_sec,
            attn_implementation=attn_implementation,
            dtype=self._dtype_name(self.torch_dtype),
            device=runtime_device,
            runtime_quantization=runtime_quantization,
            low_cpu_mem_usage=self.kwargs.get("low_cpu_mem_usage", True),
        )

    @staticmethod
    def _dtype_name(dtype: torch.dtype | str) -> str:
        if isinstance(dtype, str):
            return dtype

        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
            torch.float32: "float32",
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported OpenVLA runtime dtype: {dtype}")
        return mapping[dtype]

    def load(self) -> None:
        logger.info("Starting OpenVLA subprocess runtime from %s", self.model_path)
        self._runtime.load()
        self.model = self._runtime
        self.processor = True

    def close(self) -> None:
        self._runtime.close()
        self.model = None
        self.processor = None

    def _predict_request(
        self,
        *,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None = None,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not loaded. Call load() first.")

        final_unnorm_key = unnorm_key if unnorm_key is not None else self.default_unnorm_key
        return self._runtime.predict_action(
            image=image,
            instruction=instruction,
            unnorm_key=final_unnorm_key,
            extra_predict_kwargs=predict_kwargs,
        )

    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return self._predict_request(
            image=image,
            instruction=instruction,
            unnorm_key=unnorm_key,
            **kwargs,
        )

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
