"""
Adapter wrapper registry: Similar to VLA model system.
Each RobotWrapper or SimulationWrapper subclass should register itself here.
Server or launch code can lookup and instantiate by name.
"""

import logging

from .base import AdapterWrapper

logger = logging.getLogger(__name__)

_ADAPTER_REGISTRY: dict[str, type[AdapterWrapper]] = {}


def register_adapter(name: str, cls: type[AdapterWrapper]) -> None:
    _ADAPTER_REGISTRY[name] = cls


def get_registry() -> dict[str, type[AdapterWrapper]]:
    return dict(_ADAPTER_REGISTRY)


def _register_builtin_adapters() -> None:
    try:
        from .libero_adapter import LiberoAdapterWrapper
    except ImportError as err:
        logger.warning("Libero adapter not available: %s", err)
    else:
        register_adapter("libero", LiberoAdapterWrapper)

    try:
        from .calvin_adapter import CalvinAdapterWrapper
    except ImportError as err:
        logger.warning("CALVIN adapter import failed: %s", err)
    else:
        register_adapter("calvin", CalvinAdapterWrapper)


_register_builtin_adapters()
