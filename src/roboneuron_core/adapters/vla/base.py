from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

if TYPE_CHECKING:
    import torch


def _default_device() -> Any:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class ModelWrapper(ABC):
    """Base interface for VLA model wrappers."""

    def __init__(
        self,
        model_path: str | Path,
        device: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Store model location, target device, and wrapper-specific options."""
        self.model_path = str(model_path)
        self.device = device if device is not None else _default_device()
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
