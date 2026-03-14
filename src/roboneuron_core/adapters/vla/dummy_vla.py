"""Lightweight VLA wrapper for end-to-end pipeline validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn as nn

from .base import ModelWrapper

logger = logging.getLogger(__name__)

DUMMY_MODEL_PATH = "__dummy_vla__"


class DummyVLAModel(nn.Module):
    """Small deterministic network that mimics the OpenVLA call shape."""

    def __init__(self, img_size: int = 64, action_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.img_size = img_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.current_prune_ratio = 0.0
        self.last_fastv_config: dict[str, Any] | None = None

        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        conv_out_size = (img_size // 4) * (img_size // 4) * 32
        self.mlp = nn.Sequential(
            nn.Linear(conv_out_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.constant_(module.weight, 0.01)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.constant_(module.weight, 0.005)
                nn.init.zeros_(module.bias)

    def apply_pruning(self, layer_name: str, prune_ratio: float) -> None:
        self.current_prune_ratio = prune_ratio
        logger.info(
            "[DummyPrune] Pretend pruning layer '%s' by %.1f%%",
            layer_name,
            prune_ratio * 100.0,
        )

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        encoded = self.conv(img_tensor)
        flattened = encoded.flatten(1)
        return self.mlp(flattened)

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        fastv_config: dict[str, Any] | None = None,
        prune_config: dict[str, Any] | None = None,
        **_: Any,
    ) -> np.ndarray:
        if prune_config is not None:
            layer = str(prune_config.get("layer", "dummy_layer"))
            ratio = float(prune_config.get("ratio", 0.5))
            self.apply_pruning(layer, ratio)

        self.last_fastv_config = dict(fastv_config) if fastv_config is not None else None
        if self.last_fastv_config is not None:
            logger.info(
                "[DummyFastV] use_fastv=%s, k=%s, r=%s",
                self.last_fastv_config.get("use_fastv", False),
                self.last_fastv_config.get("fastv_k"),
                self.last_fastv_config.get("fastv_r"),
            )

        logits = self(pixel_values)
        return logits.cpu().numpy()


class DummyVLAWrapper(ModelWrapper):
    """Pipeline-oriented dummy VLA that supports accel/prune plumbing."""

    def __init__(
        self,
        model_path: str | Path = DUMMY_MODEL_PATH,
        device: torch.device | None = None,
        img_size: int = 64,
        action_dim: int = 7,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_path=model_path, device=device, **kwargs)
        self.img_size = img_size
        self.action_dim = action_dim
        self.model: DummyVLAModel | None = None
        self.last_accel_method: str | None = None
        self.last_accel_config: dict[str, Any] | None = None
        self.last_prune_config: dict[str, Any] | None = None

    def load(self) -> None:
        self.model = DummyVLAModel(
            img_size=self.img_size,
            action_dim=self.action_dim,
            hidden_dim=128,
        ).to(self.device)
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
        accel_method: str | None = None,
        accel_config: dict[str, Any] | None = None,
        prune_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del instruction, unnorm_key, kwargs
        if self.model is None:
            raise RuntimeError("DummyVLAWrapper.load() must be called before predict_action().")

        self.last_accel_method = accel_method
        self.last_accel_config = dict(accel_config) if accel_config is not None else None
        self.last_prune_config = dict(prune_config) if prune_config is not None else None

        img = image.convert("RGB").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)

        fastv_cfg = self.last_accel_config if accel_method == "fastv" else None
        return self.model.predict_action(
            pixel_values=tensor,
            fastv_config=fastv_cfg,
            prune_config=self.last_prune_config,
        )
