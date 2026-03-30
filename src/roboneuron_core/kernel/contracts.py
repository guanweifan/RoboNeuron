"""Kernel-level state and action contract primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL = "normalized_cartesian_velocity"
TASK_SPACE_STATE_SOURCE = "task_space_state"


def _coerce_vector(values: Any, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values, got shape {array.shape}.")
    return array


@dataclass(frozen=True)
class StateSnapshot:
    """Canonical runtime-facing state for the current execution path."""

    position_xyz: tuple[float, float, float]
    orientation_rpy: tuple[float, float, float]
    gripper_open_fraction: float
    frame: str = "base"
    source: str = TASK_SPACE_STATE_SOURCE
    captured_at_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        position = _coerce_vector(self.position_xyz, expected_size=3, name="position_xyz")
        orientation = _coerce_vector(self.orientation_rpy, expected_size=3, name="orientation_rpy")
        object.__setattr__(self, "position_xyz", tuple(float(value) for value in position))
        object.__setattr__(self, "orientation_rpy", tuple(float(value) for value in orientation))
        object.__setattr__(
            self,
            "gripper_open_fraction",
            float(np.clip(float(self.gripper_open_fraction), 0.0, 1.0)),
        )

    @classmethod
    def from_vector(
        cls,
        values: Any,
        *,
        frame: str = "base",
        source: str = TASK_SPACE_STATE_SOURCE,
        captured_at_sec: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        vector = _coerce_vector(values, expected_size=7, name="state vector")
        return cls(
            position_xyz=tuple(float(value) for value in vector[:3]),
            orientation_rpy=tuple(float(value) for value in vector[3:6]),
            gripper_open_fraction=float(vector[6]),
            frame=frame,
            source=source,
            captured_at_sec=captured_at_sec,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_pose_and_gripper(
        cls,
        position_xyz: Any,
        orientation_rpy: Any,
        gripper_open_fraction: float,
        *,
        frame: str = "base",
        source: str = TASK_SPACE_STATE_SOURCE,
        captured_at_sec: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateSnapshot:
        return cls(
            position_xyz=tuple(float(value) for value in _coerce_vector(position_xyz, expected_size=3, name="position_xyz")),
            orientation_rpy=tuple(float(value) for value in _coerce_vector(orientation_rpy, expected_size=3, name="orientation_rpy")),
            gripper_open_fraction=gripper_open_fraction,
            frame=frame,
            source=source,
            captured_at_sec=captured_at_sec,
            metadata=dict(metadata or {}),
        )

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                *self.position_xyz,
                *self.orientation_rpy,
                self.gripper_open_fraction,
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class ActionContract:
    """Description of an action transport contract accepted by the runtime."""

    transport: str
    protocol: str
    frame: str = "tool"
    action_dim: int | None = None
    chunked: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action_dim is not None and self.action_dim <= 0:
            raise ValueError(f"action_dim must be positive when provided, got {self.action_dim}.")

    @classmethod
    def raw_action_chunk(
        cls,
        *,
        protocol: str,
        frame: str = "tool",
        action_dim: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ActionContract:
        return cls(
            transport="raw_action_chunk",
            protocol=protocol,
            frame=frame,
            action_dim=action_dim,
            chunked=True,
            metadata=dict(metadata or {}),
        )

    def validate_action_matrix(self, values: Any) -> np.ndarray:
        action_np = np.asarray(values, dtype=np.float64)
        if action_np.ndim == 1:
            action_np = action_np.reshape(1, -1)
        if action_np.ndim != 2 or action_np.shape[1] == 0:
            raise ValueError(f"Expected an action matrix with shape (T, D), got {action_np.shape}.")
        if self.action_dim is not None and action_np.shape[1] != self.action_dim:
            raise ValueError(
                f"{self.transport} expects action_dim={self.action_dim}, got {action_np.shape[1]}."
            )
        return action_np

