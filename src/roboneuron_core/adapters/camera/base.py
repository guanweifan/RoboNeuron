"""
Abstract base class for camera wrappers.
"""
from abc import ABC, abstractmethod

import numpy as np


class CameraWrapper(ABC):
    """Abstract base class for camera wrappers."""


    def __init__(self) -> None:
        self.properties: dict[str, float | None] = {}


    @abstractmethod
    def open(self) -> None:
        pass


    @abstractmethod
    def close(self) -> None:
        pass


    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        pass


    @abstractmethod
    def is_opened(self) -> bool:
        pass