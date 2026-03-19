"""Helpers for 7D task-space state ROS messages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TASK_SPACE_STATE_TOPIC = "/task_space_state"


def _task_space_state_cls():
    from roboneuron_interfaces.msg import TaskSpaceState

    return TaskSpaceState


def array_to_task_space_state_message(state: Sequence[float] | np.ndarray) -> object:
    """Convert a 7D task-space state vector into a ``TaskSpaceState`` ROS message."""

    state_np = np.asarray(state, dtype=np.float64).reshape(-1)
    if state_np.size != 7:
        raise ValueError(f"Expected a 7D task-space state vector, got shape {state_np.shape}.")

    message = _task_space_state_cls()()
    message.x = float(state_np[0])
    message.y = float(state_np[1])
    message.z = float(state_np[2])
    message.roll = float(state_np[3])
    message.pitch = float(state_np[4])
    message.yaw = float(state_np[5])
    message.gripper_open_fraction = float(state_np[6])
    return message


def task_space_state_message_to_array(message: object) -> np.ndarray:
    """Convert a ``TaskSpaceState`` ROS message into a 7D state vector."""

    return np.array(
        [
            getattr(message, "x"),
            getattr(message, "y"),
            getattr(message, "z"),
            getattr(message, "roll"),
            getattr(message, "pitch"),
            getattr(message, "yaw"),
            getattr(message, "gripper_open_fraction"),
        ],
        dtype=np.float64,
    )
