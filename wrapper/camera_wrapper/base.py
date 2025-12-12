"""
Abstract base class for camera wrappers.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
import numpy as np


class CameraWrapper(ABC):
    """Abstract base class for camera wrappers."""


    def __init__(self) -> None:
        self.properties: Dict[str, Optional[float]] = {}


    @abstractmethod
    def open(self) -> None:
        pass


    @abstractmethod
    def close(self) -> None:
        pass


    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass


    @abstractmethod
    def is_opened(self) -> bool:
        pass