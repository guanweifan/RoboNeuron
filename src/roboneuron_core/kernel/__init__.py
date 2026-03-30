"""Kernel primitives shared across RoboNeuron core runtime modules."""

from .contracts import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    TASK_SPACE_STATE_SOURCE,
    ActionContract,
    StateSnapshot,
)
from .health import HealthLevel, HealthStatus
from .profile import RuntimeProfile
from .session import ExecutionSession, ExecutionSessionStatus, ExecutionTrace

__all__ = [
    "ActionContract",
    "DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL",
    "ExecutionSession",
    "ExecutionSessionStatus",
    "ExecutionTrace",
    "HealthLevel",
    "HealthStatus",
    "RuntimeProfile",
    "StateSnapshot",
    "TASK_SPACE_STATE_SOURCE",
]
