from __future__ import annotations

import sys
import types

import numpy as np


def _install_fake_task_space_state_module() -> None:
    fake_roboneuron_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_roboneuron_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")

    class FakeTaskSpaceState:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.gripper_open_fraction = 0.0

    fake_roboneuron_interfaces_msg.TaskSpaceState = FakeTaskSpaceState
    fake_roboneuron_interfaces.msg = fake_roboneuron_interfaces_msg

    sys.modules["roboneuron_interfaces"] = fake_roboneuron_interfaces
    sys.modules["roboneuron_interfaces.msg"] = fake_roboneuron_interfaces_msg


def test_task_space_state_round_trip() -> None:
    _install_fake_task_space_state_module()

    from roboneuron_core.utils.task_space_state import (
        array_to_task_space_state_message,
        task_space_state_message_to_array,
    )

    state = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.75], dtype=np.float64)

    message = array_to_task_space_state_message(state)
    decoded = task_space_state_message_to_array(message)

    np.testing.assert_allclose(decoded, state)
