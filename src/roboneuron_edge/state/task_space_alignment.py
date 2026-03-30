"""Edge-owned task-space alignment helpers."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from roboneuron_core.kernel import StateSnapshot


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


def rotation_matrix_to_rpy(rotation_matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into roll/pitch/yaw."""

    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    sin_pitch = float(-matrix[2, 0])
    pitch = math.asin(np.clip(sin_pitch, -1.0, 1.0))
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-6:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0

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

    roll_pitch_yaw = quaternion_xyzw_to_rpy(orientation_xyzw)
    snapshot = StateSnapshot.from_pose_and_gripper(
        position_xyz,
        roll_pitch_yaw,
        gripper_open_fraction,
    )
    return snapshot.as_vector()


def pose_matrix_to_state_vector(
    pose_matrix: Sequence[Sequence[float]] | np.ndarray,
    gripper_open_fraction: float,
) -> np.ndarray:
    """Build the canonical 7D task-space state vector from a 4x4 pose matrix."""

    transform = np.asarray(pose_matrix, dtype=np.float64).reshape(4, 4)
    snapshot = StateSnapshot.from_pose_and_gripper(
        transform[:3, 3],
        rotation_matrix_to_rpy(transform[:3, :3]),
        gripper_open_fraction,
    )
    return snapshot.as_vector()
