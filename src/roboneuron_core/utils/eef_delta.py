"""Helpers for the unified end-effector delta command message."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from roboneuron_interfaces.msg import EEFDeltaCommand

EEF_DELTA_CMD_TOPIC = "/eef_delta_cmd"


def array_to_eef_delta_command(action: Sequence[float] | np.ndarray) -> EEFDeltaCommand:
    """Convert a 7D action vector into an ``EEFDeltaCommand`` ROS message."""
    action_np = np.asarray(action, dtype=np.float64).reshape(-1)
    if action_np.size != 7:
        raise ValueError(f"Expected a 7D action vector, got shape {action_np.shape}.")

    message = EEFDeltaCommand()
    message.delta_x = float(action_np[0])
    message.delta_y = float(action_np[1])
    message.delta_z = float(action_np[2])
    message.delta_roll = float(action_np[3])
    message.delta_pitch = float(action_np[4])
    message.delta_yaw = float(action_np[5])
    message.gripper_cmd = float(action_np[6])
    return message


def eef_delta_command_to_array(message: EEFDeltaCommand) -> np.ndarray:
    """Convert an ``EEFDeltaCommand`` ROS message into a 7D action vector."""
    return np.array(
        [
            message.delta_x,
            message.delta_y,
            message.delta_z,
            message.delta_roll,
            message.delta_pitch,
            message.delta_yaw,
            message.gripper_cmd,
        ],
        dtype=np.float64,
    )
