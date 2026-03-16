from __future__ import annotations

import logging
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from roboneuron_core.runtime.openvla_oft_client import OpenVLAOFTSubprocessClient

from .base import ModelWrapper

logger = logging.getLogger(__name__)
_ORCHESTRATION_ONLY_PREDICT_KWARGS = {
    "accel_method",
    "accel_level",
    "accel_config",
    "prune_config",
}


class OpenVLAOFTWrapper(ModelWrapper):
    """Thin adapter that proxies OpenVLA-OFT inference to a dedicated subprocess runtime."""

    def __init__(
        self,
        model_path: str | Path,
        attn_implementation: str | None = None,
        dtype: torch.dtype | str | None = None,
        default_unnorm_key: str | None = None,
        runtime_python: str | Path | None = None,
        runtime_module: str = "roboneuron_core.runtime.openvla_oft_worker",
        runtime_extra_python_paths: list[str] | tuple[str, ...] | None = None,
        runtime_startup_timeout_sec: float = 1800.0,
        runtime_request_timeout_sec: float = 300.0,
        runtime_device: str = "auto",
        use_l1_regression: bool | None = None,
        use_diffusion: bool | None = None,
        use_film: bool | None = None,
        use_proprio: bool | None = None,
        num_images_in_input: int | None = None,
        num_diffusion_steps_inference: int = 50,
        lora_rank: int = 32,
        center_crop: bool = True,
        robot_platform: str | None = None,
        default_proprio: list[float] | tuple[float, ...] | np.ndarray | None = None,
        base_model_path: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path, **kwargs)
        self.attn_implementation = attn_implementation
        self.default_unnorm_key = default_unnorm_key
        self.torch_dtype = dtype or torch.bfloat16
        self._runtime = OpenVLAOFTSubprocessClient(
            model_path=self.model_path,
            runtime_python=runtime_python,
            runtime_module=runtime_module,
            runtime_extra_python_paths=runtime_extra_python_paths,
            startup_timeout_sec=runtime_startup_timeout_sec,
            request_timeout_sec=runtime_request_timeout_sec,
            attn_implementation=attn_implementation,
            dtype=self._dtype_name(self.torch_dtype),
            device=runtime_device,
            low_cpu_mem_usage=self.kwargs.get("low_cpu_mem_usage", True),
            use_l1_regression=use_l1_regression,
            use_diffusion=use_diffusion,
            use_film=use_film,
            use_proprio=use_proprio,
            num_images_in_input=num_images_in_input,
            num_diffusion_steps_inference=num_diffusion_steps_inference,
            lora_rank=lora_rank,
            center_crop=center_crop,
            unnorm_key=default_unnorm_key,
            robot_platform=robot_platform,
            default_proprio=default_proprio,
            base_model_path=base_model_path,
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
            raise ValueError(f"Unsupported OpenVLA-OFT runtime dtype: {dtype}")
        return mapping[dtype]

    def load(self) -> None:
        logger.info("Starting OpenVLA-OFT subprocess runtime from %s", self.model_path)
        self._runtime.load()
        self.model = self._runtime
        self.processor = True

    def close(self) -> None:
        self._runtime.close()
        self.model = None
        self.processor = None

    def _build_observation(
        self,
        image: Image.Image | list[Image.Image] | dict[str, Any],
        instruction: str,
        **predict_kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        runtime_predict_kwargs = {
            key: value
            for key, value in predict_kwargs.items()
            if key not in _ORCHESTRATION_ONLY_PREDICT_KWARGS
        }

        if isinstance(image, dict):
            observation = dict(image)
        else:
            images = list(image) if isinstance(image, list) else [image]
            observation: dict[str, Any] = {}
            if images:
                observation["full_image"] = images[0]
            if len(images) == 2:
                observation["wrist_image"] = images[1]
            elif len(images) >= 3:
                observation["left_wrist_image"] = images[1]
                observation["right_wrist_image"] = images[2]
                if len(images) > 3:
                    observation["images"] = images

        if "instruction" not in observation:
            observation["instruction"] = instruction

        for source_key, target_key in (("proprio", "state"), ("state", "state")):
            if source_key in runtime_predict_kwargs and "state" not in observation:
                observation[target_key] = runtime_predict_kwargs.pop(source_key)

        if "images" not in observation and "wrist_images" in runtime_predict_kwargs:
            wrist_images = list(runtime_predict_kwargs.pop("wrist_images"))
            full_image = observation.pop("full_image", None)
            if full_image is not None:
                observation["images"] = [full_image, *wrist_images]

        return observation, runtime_predict_kwargs

    def _predict_request(
        self,
        *,
        image: Image.Image | list[Image.Image] | dict[str, Any],
        instruction: str,
        unnorm_key: str | None = None,
        **predict_kwargs: Any,
    ) -> np.ndarray:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not loaded. Call load() first.")

        final_unnorm_key = unnorm_key if unnorm_key is not None else self.default_unnorm_key
        observation, runtime_predict_kwargs = self._build_observation(
            image=image,
            instruction=instruction,
            **predict_kwargs,
        )
        return self._runtime.predict_action(
            observation=observation,
            instruction=instruction,
            unnorm_key=final_unnorm_key,
            extra_predict_kwargs=runtime_predict_kwargs,
        )

    def predict_action(
        self,
        image: Image.Image | list[Image.Image] | dict[str, Any],
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
