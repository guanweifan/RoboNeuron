"""
Robot adapter registry.

Server code looks up robot adapters by name through this module.
"""

from .base import AdapterWrapper

_ADAPTER_REGISTRY: dict[str, type[AdapterWrapper]] = {}


def register_adapter(name: str, cls: type[AdapterWrapper]) -> None:
    _ADAPTER_REGISTRY[name] = cls


def get_registry() -> dict[str, type[AdapterWrapper]]:
    return dict(_ADAPTER_REGISTRY)


def _register_builtin_adapters() -> None:
    from .dummy_robot import DummyRobotAdapterWrapper

    register_adapter("dummy", DummyRobotAdapterWrapper)


_register_builtin_adapters()
