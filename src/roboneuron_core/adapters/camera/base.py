"""Base interface for camera wrappers."""
from abc import ABC, abstractmethod

import numpy as np


class CameraWrapper(ABC):
    """Abstract base class for camera wrappers."""

    def __init__(self) -> None:
        self.properties: dict[str, float | None] = {}

    @abstractmethod
    def open(self) -> None:
        """Open the camera stream."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the camera stream."""
        pass

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame from the camera."""
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        """Return whether the camera stream is active."""
        pass
