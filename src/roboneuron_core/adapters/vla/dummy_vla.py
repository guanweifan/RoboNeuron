"""Lightweight torch-free VLA wrapper for end-to-end pipeline validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .base import ModelWrapper

logger = logging.getLogger(__name__)

DUMMY_MODEL_PATH = "__dummy_vla__"


class DummyVLAModel:
    """Small deterministic numpy model that mimics the OpenVLA call shape."""

    def __init__(self, img_size: int = 64, action_dim: int = 7) -> None:
        self.img_size = img_size
        self.action_dim = action_dim
        self._offsets = np.linspace(-0.35, 0.35, action_dim, dtype=np.float32)

    def predict_action(
        self,
        pixel_values: np.ndarray,
        **_: Any,
    ) -> np.ndarray:
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected a 4D image batch, got shape {pixel_values.shape}.")

        channel_means = pixel_values.mean(axis=(2, 3))
        base_signal = channel_means.mean(axis=1, keepdims=True)
        contrast_signal = (channel_means[:, 1:2] - channel_means[:, :1]) * 0.25

        action = np.repeat(base_signal, self.action_dim, axis=1) + self._offsets
        action[:, 0:1] += contrast_signal
        action[:, -1:] = np.clip(base_signal, 0.0, 1.0)
        return np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)


class DummyVLAWrapper(ModelWrapper):
    """Pipeline-oriented dummy VLA for local validation."""

    def __init__(
        self,
        model_path: str | Path = DUMMY_MODEL_PATH,
        device: Any | None = None,
        img_size: int = 64,
        action_dim: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, device=device, **kwargs)
        self.img_size = img_size
        self.action_dim = action_dim
        self.model: DummyVLAModel | None = None

    def load(self) -> None:
        self.model = DummyVLAModel(
            img_size=self.img_size,
            action_dim=self.action_dim,
        )
        self.processor = True
        logger.info(
            "Loaded DummyVLAWrapper on device=%s img_size=%s action_dim=%s",
            self.device,
            self.img_size,
            self.action_dim,
        )

    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        del instruction, unnorm_key, kwargs
        if self.model is None:
            raise RuntimeError("DummyVLAWrapper.load() must be called before predict_action().")

        img = image.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pixel_values = np.transpose(arr, (2, 0, 1))[None, ...]
        return self.model.predict_action(pixel_values=pixel_values)
