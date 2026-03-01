"""
openvla_oft_wrapper.py
"""
from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# Import VLA specific utilities
# Ensure these paths match your actual project structure
from third_party.vla_src.openvla_oft.experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from third_party.vla_src.openvla_oft.experiments.robot.robot_utils import get_image_resize_size
from third_party.vla_src.openvla_oft.prismatic.vla.constants import PROPRIO_DIM

# Assumed location of the abstract base class provided in your prompt
from .base import ModelWrapper


class OpenVLAOFTWrapper(ModelWrapper):
    """
    Implementation of the VLA wrapper for the OpenVLA-OFT model family.
    Encapsulates configuration, loading, and inference logic.
    """

    @dataclass
    class Config:
        """
        Internal configuration for OpenVLA model parameters.
        Defaults match the original DeployConfig.
        """
        model_family: str = "openvla"
        pretrained_checkpoint: str | Path | None = None
        
        use_l1_regression: bool = True
        use_diffusion: bool = False
        use_film: bool = True
        use_proprio: bool = True
        use_relative_actions: bool = False
        
        num_images_in_input: int = 2
        lora_rank: int = 32
        num_diffusion_steps_inference: int = 50
        
        center_crop: bool = True
        unnorm_key: str | Path = "franka_kitchen"
        
        load_in_8bit: bool = False
        load_in_4bit: bool = False

    def __init__(self, model_path: str | Path, device: torch.device | None = None, **kwargs):
        super().__init__(model_path, device=device, **kwargs)
        
        # Initialize config: allow override via kwargs, otherwise use defaults
        self.cfg = self.Config(**{k: v for k, v in kwargs.items() if k in self.Config.__annotations__})
        
        # Override checkpoint path if explicitly provided in __init__ arguments
        if model_path:
            self.cfg.pretrained_checkpoint = str(model_path)

        # Placeholders
        self.vla = None
        self.proprio_projector = None
        self.action_head = None
        self.resize_size = None

    def load(self) -> None:
        """
        Loads the VLA model, processor, action head, and projectors onto the device.
        """
        logging.info(f"Loading OpenVLA-OFT model from: {self.cfg.pretrained_checkpoint}")

        self.vla = get_vla(self.cfg)

        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, self.vla.llm_dim, PROPRIO_DIM
            )

        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = get_action_head(self.cfg, self.vla.llm_dim)

        self.processor = get_processor(self.cfg)

        self.resize_size = get_image_resize_size(self.cfg)

        if self.device.type != "cpu":
            self.vla.to(self.device)
            if self.proprio_projector:
                self.proprio_projector.to(self.device)
            if self.action_head:
                self.action_head.to(self.device)

        self.model = self.vla
        logging.info("OpenVLA-OFT model loaded successfully.")

    def _prepare_image(self, image: Image.Image) -> np.ndarray:
        """
        Resizes and converts PIL Image to (H, W, C) numpy array (uint8).
        """
        if self.resize_size is not None:
            # resize expects (width, height)
            image = image.resize((self.resize_size[1], self.resize_size[0]))
        
        arr = np.asarray(image)
        
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            arr = arr[..., :3]
            
        return arr

    def predict_action(
        self, 
        image: Image.Image | list[Image.Image] | dict[str, Any], 
        instruction: str | None = None, 
        unnorm_key: str | None = None, 
        **kwargs
    ) -> Any:
        """
        Performs inference to predict robot actions.
        
        Args:
            image: Single PIL Image, List of PIL Images, or pre-formed observation dict.
            instruction: Natural language instruction.
            unnorm_key: Specific key for action un-normalization.
            **kwargs: Can include 'proprio' (np.ndarray) for robot state.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Please call load() first.")

        observation: dict[str, Any] = {}
        
        if isinstance(image, dict):
            observation = image
            if "instruction" not in observation and instruction:
                observation["instruction"] = instruction
        else:
            images_list = image if isinstance(image, list) else [image]
            
            if len(images_list) != self.cfg.num_images_in_input:
                logging.warning(
                    f"Expected {self.cfg.num_images_in_input} images, got {len(images_list)}."
                )

            observation["images"] = [self._prepare_image(im) for im in images_list]
            
            if instruction:
                observation["instruction"] = instruction
            
            proprio = kwargs.get("proprio")
            if proprio is not None:
                observation["proprio"] = proprio

        final_unnorm_key = unnorm_key or self.cfg.unnorm_key
        if not final_unnorm_key:
             raise ValueError("No unnorm_key provided in config or arguments.")

        try:
            action = get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                observation,
                observation.get("instruction", ""),
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
            )
            return action
            
        except Exception as err:
            logging.error(traceback.format_exc())
            raise RuntimeError("Failed to predict action.") from err
