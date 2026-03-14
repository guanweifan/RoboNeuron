"""VLA model wrapper registry."""

from __future__ import annotations

import logging

from .base import ModelWrapper

logger = logging.getLogger(__name__)

_MODEL_REGISTRY: dict[str, type[ModelWrapper]] = {}


def register_model(name: str, cls: type[ModelWrapper]) -> None:
    _MODEL_REGISTRY[name] = cls


def get_registry() -> dict[str, type[ModelWrapper]]:
    return dict(_MODEL_REGISTRY)


def _register_builtin_models() -> None:
    from .dummy_vla import DummyVLAWrapper

    register_model("dummy", DummyVLAWrapper)

    try:
        from .openvla import OpenVLAWrapper
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("OpenVLA wrapper not available: %s", exc)
    else:
        register_model("openvla", OpenVLAWrapper)

    try:
        from .openvla_oft import OpenVLAOFTWrapper
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("OpenVLA-OFT wrapper not available: %s", exc)
    else:
        register_model("openvla-oft", OpenVLAOFTWrapper)

    try:
        from .openvla_fastv import OpenVLAFastVWrapper
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("OpenVLA-FastV wrapper not available: %s", exc)
    else:
        register_model("openvla-fastv", OpenVLAFastVWrapper)
        register_model("openvla_fastv", OpenVLAFastVWrapper)

    try:
        from .pi0 import Pi0Wrapper
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("Pi0 wrapper not available: %s", exc)
    else:
        register_model("pi0", Pi0Wrapper)


_register_builtin_models()
