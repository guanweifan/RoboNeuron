# wrappers/dummy_vla.py
import logging
from pathlib import Path
from typing import Union, Optional, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ModelWrapper  

logger = logging.getLogger("wrappers.dummy_vla")


class DummyVLAModel(nn.Module):
    """
    A minimal VLA-like model used to validate:
    - load() behavior
    - prune / FastV config plumbing
    - predict_action() interface wiring

    It does not load large checkpoints and only runs lightweight layers.
    """

    def __init__(self, img_size: int = 64, action_dim: int = 7, hidden_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # A very simple vision encoder: Conv + MLP
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

        self.current_prune_ratio = 0.0

    # ---- Simulated pruning ----
    def apply_pruning(self, layer_name: str, prune_ratio: float):
        self.current_prune_ratio = prune_ratio
        print(
            f"[DummyPrune] Pretend pruning layer '{layer_name}' by {prune_ratio * 100:.1f}%"
        )

    # ---- Forward + predict_action ----
    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        x = self.conv(img_tensor)         
        x = x.flatten(1)                  
        x = self.mlp(x)                  
        return x

    @torch.no_grad()
    def predict_action(
        self,
        pixel_values: torch.Tensor,
        fastv_config: Optional[dict] = None,
        prune_config: Optional[dict] = None,
        **kwargs,
    ):
        """
        Simulate OpenVLA's predict_action:
        - Accept fastv_config / prune_config
        - Log incoming config values
        - Forward and return an action vector as numpy
        """
        # 1) Handle prune config
        if prune_config is not None:
            layer = prune_config.get("layer", "dummy_layer")
            ratio = float(prune_config.get("ratio", 0.5))
            self.apply_pruning(layer, ratio)


        if fastv_config is not None:
            use_fastv = fastv_config.get("use_fastv", False)
            k = fastv_config.get("fastv_k", None)
            r = fastv_config.get("fastv_r", None)
            print(f"[DummyFastV] use_fastv={use_fastv}, k={k}, r={r}")


        logits = self(pixel_values)
        return logits.cpu().numpy()
    


class TestVLAWrapper(ModelWrapper):
    """
    Lightweight wrapper used to replace OpenVLA for pipeline tests.

    - load(): build DummyVLAModel without loading HF weights
    - predict_action(): convert PIL.Image + instruction to tensor and call DummyVLAModel.predict_action
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[torch.device] = None,
        img_size: int = 64,
        action_dim: int = 7,
        **kwargs,
    ):
        super().__init__(model_path=model_path, device=device, **kwargs)
        self.img_size = img_size
        self.action_dim = action_dim
        self.model: Optional[DummyVLAModel] = None

    def load(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.model = DummyVLAModel(
            img_size=self.img_size,
            action_dim=self.action_dim,
            hidden_dim=128,
        ).to(self.device)

        logger.info(
            f"Loaded DummyVLAModel on device={self.device}, img_size={self.img_size}, "
            f"action_dim={self.action_dim}"
        )

    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        accel_method: Optional[str] = None,
        accel_config: Optional[dict] = None,
        prune_config: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        if self.model is None:
            raise RuntimeError("TestVLAWrapper.load() must be called before predict_action().")
    
        # Print args from infer_mcp for quick inspection
        print(f"[TestVLAWrapper] accel_method={accel_method}")
        print(f"[TestVLAWrapper] accel_config={accel_config}")
        print(f"[TestVLAWrapper] prune_config={prune_config}")
    
        # Convert image to tensor
        img = image.resize((self.img_size, self.img_size))
        arr = np.array(img).astype("float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0).to(self.device)
    
        # If accel_method is fastv, pass accel_config as fastv_config
        fastv_cfg = accel_config if accel_method == "fastv" else None
    
        action = self.model.predict_action(
            pixel_values=tensor,
            fastv_config=fastv_cfg,
            prune_config=prune_config,
        )
        return action
    
