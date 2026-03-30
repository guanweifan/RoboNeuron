"""Helpers for 7D task-space state ROS messages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from roboneuron_core.kernel.contracts import TASK_SPACE_STATE_SOURCE, StateSnapshot

TASK_SPACE_STATE_TOPIC = "/task_space_state"


def _task_space_state_cls():
    from roboneuron_interfaces.msg import TaskSpaceState

    return TaskSpaceState


def array_to_task_space_state_message(state: Sequence[float] | np.ndarray) -> object:
    """Convert a 7D task-space state vector into a ``TaskSpaceState`` ROS message."""

    snapshot = StateSnapshot.from_vector(state)
    return state_snapshot_to_task_space_state_message(snapshot)


def state_snapshot_to_task_space_state_message(snapshot: StateSnapshot) -> object:
    """Convert a ``StateSnapshot`` into a ``TaskSpaceState`` ROS message."""

    message = _task_space_state_cls()()
    state_np = snapshot.as_vector()
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

    return task_space_state_message_to_state_snapshot(message).as_vector()


def task_space_state_message_to_state_snapshot(
    message: object,
    *,
    frame: str = "base",
    source: str = TASK_SPACE_STATE_SOURCE,
) -> StateSnapshot:
    """Convert a ``TaskSpaceState`` ROS message into a ``StateSnapshot``."""

    return StateSnapshot.from_vector(
        [
            message.x,
            message.y,
            message.z,
            message.roll,
            message.pitch,
            message.yaw,
            message.gripper_open_fraction,
        ],
        frame=frame,
        source=source,
    )
