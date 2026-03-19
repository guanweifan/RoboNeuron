"""Helpers for 7D task-space state ROS messages."""

from __future__ import annotations

import math
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
            message.x,
            message.y,
            message.z,
            message.roll,
            message.pitch,
            message.yaw,
            message.gripper_open_fraction,
        ],
        dtype=np.float64,
    )


def quaternion_xyzw_to_rpy(quaternion: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert an ``(x, y, z, w)`` quaternion to roll/pitch/yaw."""

    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


def gripper_joint_positions_to_open_fraction(
    joint_positions: Sequence[float] | np.ndarray,
    *,
    closed_position: float = 0.0,
    open_position: float = 0.04,
) -> float:
    """Normalize one or more finger joint positions into ``[0, 1]``."""

    positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if positions.size == 0:
        raise ValueError("At least one gripper joint position is required.")

    span = float(open_position - closed_position)
    if abs(span) < 1e-9:
        raise ValueError("open_position and closed_position must differ.")

    mean_position = float(np.mean(positions))
    open_fraction = (mean_position - closed_position) / span
    return float(np.clip(open_fraction, 0.0, 1.0))


def extract_gripper_open_fraction_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    *,
    joint_names: Sequence[str] | None = None,
    closed_position: float = 0.0,
    open_position: float = 0.04,
) -> float:
    """Extract normalized gripper openness from a ``JointState`` payload."""

    if joint_names is None:
        selected_positions = [
            float(position)
            for name, position in zip(names, positions, strict=False)
            if "finger" in name.lower()
        ]
    else:
        position_by_name = {
            name: float(position)
            for name, position in zip(names, positions, strict=False)
        }
        selected_positions = [position_by_name[name] for name in joint_names if name in position_by_name]

    if not selected_positions:
        raise ValueError("No gripper joint positions were found in the provided JointState payload.")

    return gripper_joint_positions_to_open_fraction(
        selected_positions,
        closed_position=closed_position,
        open_position=open_position,
    )


def pose_and_gripper_to_state_vector(
    position_xyz: Sequence[float] | np.ndarray,
    orientation_xyzw: Sequence[float] | np.ndarray,
    gripper_open_fraction: float,
) -> np.ndarray:
    """Build the canonical 7D task-space state vector from pose + gripper state."""

    position = np.asarray(position_xyz, dtype=np.float64).reshape(3)
    roll_pitch_yaw = quaternion_xyzw_to_rpy(orientation_xyzw)
    clipped_gripper = float(np.clip(gripper_open_fraction, 0.0, 1.0))
    return np.concatenate([position, roll_pitch_yaw, np.array([clipped_gripper], dtype=np.float64)])
