"""
Camera wrapper registry.

Server code looks up camera wrappers by name through this module.
"""

from __future__ import annotations

from .base import CameraWrapper

_CAMERA_REGISTRY: dict[str, type[CameraWrapper]] = {}


def register_camera(name: str, cls: type[CameraWrapper]) -> None:
    _CAMERA_REGISTRY[name] = cls


def get_registry() -> dict[str, type[CameraWrapper]]:
    return dict(_CAMERA_REGISTRY)


def _register_builtin_cameras() -> None:
    from .dummy_camera import DummyCameraWrapper

    register_camera("dummy", DummyCameraWrapper)


_register_builtin_cameras()
