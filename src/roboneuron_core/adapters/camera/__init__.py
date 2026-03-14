"""
Camera wrapper registry: Similar to VLA model system.
Each CameraWrapper subclass should register itself here.
Server or launch code can lookup and instantiate by name.
"""

from __future__ import annotations

import logging

from .base import CameraWrapper

logger = logging.getLogger(__name__)

_CAMERA_REGISTRY: dict[str, type[CameraWrapper]] = {}


def register_camera(name: str, cls: type[CameraWrapper]) -> None:
    _CAMERA_REGISTRY[name] = cls


def get_registry() -> dict[str, type[CameraWrapper]]:
    return dict(_CAMERA_REGISTRY)


def _register_builtin_cameras() -> None:
    from .dummy_camera import DummyCameraWrapper

    register_camera("dummy", DummyCameraWrapper)

    try:
        from .realsense import RealSenseWrapper
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        logger.warning("RealSense wrapper not available: %s", exc)
    else:
        register_camera("realsense", RealSenseWrapper)


_register_builtin_cameras()
