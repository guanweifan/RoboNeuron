"""Synthetic camera wrapper used for pipeline tests."""

from __future__ import annotations

import numpy as np

from .base import CameraWrapper


class DummyCameraWrapper(CameraWrapper):
    """Generate deterministic BGR frames without touching real camera hardware."""

    def __init__(self, width: int = 256, height: int = 256) -> None:
        super().__init__()
        self.width = width
        self.height = height
        self._open = False

    def open(self) -> None:
        self._open = True
        self.properties["width"] = float(self.width)
        self.properties["height"] = float(self.height)
        self.properties["fps"] = None

    def close(self) -> None:
        self._open = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._open:
            return False, None

        gradient_x = np.linspace(0, 255, self.width, dtype=np.uint8)[None, :]
        gradient_y = np.linspace(0, 255, self.height, dtype=np.uint8)[:, None]
        blue = np.repeat(gradient_x, self.height, axis=0)
        green = np.repeat(np.flip(gradient_x, axis=1), self.height, axis=0)
        red = np.repeat(gradient_y, self.width, axis=1)
        frame = np.stack([blue, green, red], axis=2)
        return True, frame

    def is_opened(self) -> bool:
        return self._open
