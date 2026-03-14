"""Typed models for control-domain configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ControlConfig:
    urdf_path: str
    cartesian_cmd_topic: str = "/eef_delta_cmd"
    state_feedback_topic: str = "/isaac_joint_states"
    joint_cmd_topic: str = "/isaac_joint_commands"
    cmd_msg_type: Literal["JointTrajectory", "JointState"] = "JointState"
