"""
Dummy camera wrapper for testing purposes.
"""
import numpy as np

from .base import CameraWrapper


class DummyCameraWrapper(CameraWrapper):
    def __init__(self, width: int = 256, height: int = 256):
        super().__init__()
        self.width = width
        self.height = height
        self._open = False
        self._frame_idx = 0


    def open(self) -> None:
        self._open = True
        self.properties['width'] = self.width
        self.properties['height'] = self.height
        self.properties['fps'] = None


    def close(self) -> None:
        self._open = False


    def read(self):
        if not self._open:
            return False, None
        h, w = self.height, self.width
        gradient = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        img = np.repeat(gradient, h, axis=0)
        img_bgr = np.stack([img, np.flip(img, axis=1), 255 - img], axis=2)
        self._frame_idx += 1
        return True, img_bgr


    def is_opened(self) -> bool:
        return self._open
