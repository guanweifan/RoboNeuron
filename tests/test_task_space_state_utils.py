from __future__ import annotations

import sys
import types

import numpy as np


def _install_fake_task_space_state_module(monkeypatch) -> None:
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

    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_roboneuron_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_roboneuron_interfaces_msg)


def test_task_space_state_round_trip(monkeypatch) -> None:
    _install_fake_task_space_state_module(monkeypatch)

    from roboneuron_core.utils.task_space_state import (
        array_to_task_space_state_message,
        task_space_state_message_to_array,
    )

    state = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.75], dtype=np.float64)

    message = array_to_task_space_state_message(state)
    decoded = task_space_state_message_to_array(message)

    np.testing.assert_allclose(decoded, state)


def test_quaternion_xyzw_to_rpy_identity(monkeypatch) -> None:
    _install_fake_task_space_state_module(monkeypatch)

    from roboneuron_core.utils.task_space_state import quaternion_xyzw_to_rpy

    rpy = quaternion_xyzw_to_rpy([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(rpy, np.zeros((3,), dtype=np.float64))


def test_extract_gripper_open_fraction_from_joint_state_uses_named_fingers(monkeypatch) -> None:
    _install_fake_task_space_state_module(monkeypatch)

    from roboneuron_core.utils.task_space_state import (
        extract_gripper_open_fraction_from_joint_state,
    )

    open_fraction = extract_gripper_open_fraction_from_joint_state(
        names=["fr3_joint1", "fr3_finger_joint1", "fr3_finger_joint2"],
        positions=[0.0, 0.02, 0.02],
        joint_names=["fr3_finger_joint1", "fr3_finger_joint2"],
        closed_position=0.0,
        open_position=0.04,
    )

    assert open_fraction == 0.5


def test_pose_and_gripper_to_state_vector_combines_pose_and_gripper(monkeypatch) -> None:
    _install_fake_task_space_state_module(monkeypatch)

    from roboneuron_core.utils.task_space_state import pose_and_gripper_to_state_vector

    state = pose_and_gripper_to_state_vector(
        position_xyz=[0.3, -0.1, 0.5],
        orientation_xyzw=[0.0, 0.0, 0.0, 1.0],
        gripper_open_fraction=1.2,
    )

    np.testing.assert_allclose(state[:6], np.array([0.3, -0.1, 0.5, 0.0, 0.0, 0.0]))
    assert state[6] == 1.0
