"""
Camera wrapper registry: Similar to VLA model system.
Each CameraWrapper subclass should register itself here.
Server or launch code can lookup and instantiate by name.
"""


from typing import Dict, Type
from .base import CameraWrapper


_CAMERA_REGISTRY: Dict[str, Type[CameraWrapper]] = {}


def register_camera(name: str, cls: Type[CameraWrapper]):
    _CAMERA_REGISTRY[name] = cls


def get_registry():
    return dict(_CAMERA_REGISTRY)


# Import and register built-ins
from .dummy import DummyCameraWrapper 
from .realsense import RealSenseWrapper
register_camera("dummy", DummyCameraWrapper)
register_camera("realsense", RealSenseWrapper)