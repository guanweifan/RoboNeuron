from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class ModelWrapper(ABC):
    """Base interface for VLA model wrappers."""

    def __init__(
        self,
        model_path: str | Path,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        """Store model location, target device, and wrapper-specific options."""
        self.model_path = str(model_path)
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.kwargs = kwargs
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self) -> None:
        """Load model state and any associated processor into memory."""
        raise NotImplementedError

    @abstractmethod
    def predict_action(
        self,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run inference and return the next action."""
        raise NotImplementedError
