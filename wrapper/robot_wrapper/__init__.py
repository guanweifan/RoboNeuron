"""
Adapter wrapper registry: Similar to VLA model system.
Each RobotWrapper or SimulationWrapper subclass should register itself here.
Server or launch code can lookup and instantiate by name.
"""


from typing import Dict, Type
from .base import AdapterWrapper


_ADAPTER_REGISTRY: Dict[str, Type[AdapterWrapper]] = {}

def register_adapter(name: str, cls: Type[AdapterWrapper]):
    _ADAPTER_REGISTRY[name] = cls

def get_registry():
    return dict(_ADAPTER_REGISTRY)

# Import and register built-ins
from .libero_adapter import LiberoAdapterWrapper  # noqa: E402
register_adapter("libero", LiberoAdapterWrapper)