"""Franka backend metadata used by edge-side control runtime setup."""

from __future__ import annotations

FRANKA_VENDOR_STACK = ("franka_ros2", "libfranka")
_FRANKA_ROBOT_PROFILES = {"fr3_real"}


def backend_metadata_for_robot_profile(robot_profile: str | None) -> tuple[str | None, tuple[str, ...]]:
    """Resolve backend ownership metadata for a configured robot profile."""

    if robot_profile in _FRANKA_ROBOT_PROFILES:
        return "franka", FRANKA_VENDOR_STACK
    return None, ()
