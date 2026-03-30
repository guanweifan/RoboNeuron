"""Shared action semantics used across core and edge runtime modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL


def _as_vector(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError("Action vectors must not be empty.")
    return array


@dataclass(frozen=True)
class RawActionStep:
    """One model-emitted action step before semantic interpretation."""

    values: np.ndarray
    protocol: str
    frame: str = "tool"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _as_vector(self.values))


@dataclass(frozen=True)
class ActionChunk:
    """A time-ordered chunk of raw actions."""

    steps: tuple[RawActionStep, ...]
    step_duration_sec: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("Action chunks must contain at least one step.")
        if self.step_duration_sec <= 0:
            raise ValueError("Action chunk step duration must be positive.")


@dataclass(frozen=True)
class MotionIntent:
    """Canonical control intent shared across VLA/control backends."""

    mode: str
    arm: np.ndarray
    gripper_open_fraction: float | None = None
    frame: str = "tool"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _as_vector(self.arm))


@dataclass(frozen=True)
class ActuationCommand:
    """Resolved robot actuation command ready for ROS transport."""

    joint_names: list[str]
    positions: list[float]
    gripper_open_fraction: float | None = None


@dataclass(frozen=True)
class NormalizedCartesianVelocityConfig:
    """Config for normalized 7D task-space velocity actions."""

    max_linear_delta: float = 0.075
    max_rotation_delta: float = 0.15
    frame: str = "tool"
    invert_gripper: bool = False


def _require_7d_action(values: np.ndarray, protocol: str) -> np.ndarray:
    action = np.asarray(values, dtype=np.float64).reshape(-1)
    if action.size != 7:
        raise ValueError(f"{protocol} expects a 7D action vector, got shape {action.shape}.")
    return action


def _coerce_gripper_open_fraction(raw_value: float, *, invert: bool) -> float:
    value = float(raw_value)
    if value < 0.0 or value > 1.0:
        value = (np.clip(value, -1.0, 1.0) + 1.0) / 2.0
    value = float(np.clip(value, 0.0, 1.0))
    return 1.0 - value if invert else value


def motion_intent_from_eef_delta(values: np.ndarray, *, frame: str = "tool") -> MotionIntent:
    """Interpret a 7D end-effector delta command as a canonical motion intent."""

    action = _require_7d_action(values, "EEF delta")
    return MotionIntent(
        mode="cartesian_delta",
        arm=action[:6],
        gripper_open_fraction=_coerce_gripper_open_fraction(action[6], invert=False),
        frame=frame,
        metadata={"protocol": "eef_delta"},
    )


def motion_intent_from_normalized_cartesian_velocity(
    values: np.ndarray,
    *,
    config: NormalizedCartesianVelocityConfig | None = None,
    frame: str | None = None,
) -> MotionIntent:
    """Interpret normalized 7D task-space velocity commands."""

    resolved = config or NormalizedCartesianVelocityConfig()
    action = _require_7d_action(values, DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL)

    translation = np.clip(action[:3], -1.0, 1.0) * resolved.max_linear_delta
    rotation = np.clip(action[3:6], -1.0, 1.0) * resolved.max_rotation_delta
    gripper_open_fraction = _coerce_gripper_open_fraction(action[6], invert=resolved.invert_gripper)

    return MotionIntent(
        mode="cartesian_delta",
        arm=np.concatenate([translation, rotation]),
        gripper_open_fraction=gripper_open_fraction,
        frame=frame or resolved.frame,
        metadata={"protocol": DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL},
    )


def motion_intent_from_raw_step(
    step: RawActionStep,
    *,
    normalized_velocity_config: NormalizedCartesianVelocityConfig | None = None,
) -> MotionIntent:
    """Dispatch a raw action step to the correct semantic interpreter."""

    if step.protocol in {"eef_delta", "cartesian_delta"}:
        return motion_intent_from_eef_delta(step.values, frame=step.frame)

    if step.protocol == DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL:
        return motion_intent_from_normalized_cartesian_velocity(
            step.values,
            config=normalized_velocity_config,
            frame=step.frame,
        )

    raise ValueError(f"Unsupported raw action protocol: {step.protocol}")
