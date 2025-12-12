# wrappers/__init__.py

from typing import Dict, Type
from .base import ModelWrapper

_MODEL_REGISTRY: Dict[str, Type[ModelWrapper]] = {}

def register_model(name: str, cls: Type[ModelWrapper]):
    _MODEL_REGISTRY[name] = cls

def get_registry():
    return dict(_MODEL_REGISTRY)


from .openvla import OpenVLAWrapper  
from .openvla_oft import OpenVLAOFTWrapper
from .openvla_fastv import OpenVLAFastVWrapper
from .dummy import DummyVLAWrapper  

register_model("openvla", OpenVLAWrapper)
register_model("openvla-oft", OpenVLAOFTWrapper)
register_model("openvla-fastv", OpenVLAFastVWrapper)
register_model("dummy", DummyVLAWrapper)
