from __future__ import annotations

import sys
import types

import numpy as np

from roboneuron_backends.franka import FRANKA_VENDOR_STACK, backend_metadata_for_robot_profile
from roboneuron_core.kernel import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    ActuationCommand,
)
from roboneuron_core.utils.raw_action_chunk import (
    array_to_raw_action_chunk_message,
    raw_action_chunk_message_to_action_chunk,
)
from roboneuron_core.utils.task_space_state import (
    array_to_task_space_state_message,
    task_space_state_message_to_array,
)
from roboneuron_edge.runtime.control_runtime import ControlRuntime


def _install_fake_interface_messages(monkeypatch) -> None:
    fake_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")

    class FakeRawActionChunk:
        def __init__(self) -> None:
            self.protocol = ""
            self.frame = ""
            self.action_dim = 0
            self.chunk_length = 0
            self.step_duration_sec = 0.0
            self.values: list[float] = []

    class FakeTaskSpaceState:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.gripper_open_fraction = 0.0

    fake_interfaces_msg.RawActionChunk = FakeRawActionChunk
    fake_interfaces_msg.TaskSpaceState = FakeTaskSpaceState
    fake_interfaces.msg = fake_interfaces_msg

    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_interfaces_msg)


class _FakeResolver:
    def resolve(self, intent, joint_positions):  # noqa: ANN001
        del joint_positions
        return ActuationCommand(
            joint_names=["joint1"],
            positions=[float(intent.gripper_open_fraction or 0.0)],
            gripper_open_fraction=intent.gripper_open_fraction,
        )


def test_core_to_edge_split_smoke(monkeypatch) -> None:
    _install_fake_interface_messages(monkeypatch)

    backend_name, vendor_stack = backend_metadata_for_robot_profile("fr3_real")
    assert backend_name == "franka"
    assert vendor_stack == FRANKA_VENDOR_STACK

    state = np.array([0.3, -0.1, 0.5, 0.0, 0.0, 0.0, 0.75], dtype=np.float64)
    state_msg = array_to_task_space_state_message(state)
    decoded_state = task_space_state_message_to_array(state_msg)
    np.testing.assert_allclose(decoded_state, state)

    action_msg = array_to_raw_action_chunk_message(
        [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]],
        protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
        step_duration_sec=0.05,
    )
    action_chunk = raw_action_chunk_message_to_action_chunk(action_msg)

    runtime = ControlRuntime(_FakeResolver())
    runtime.queue_action_chunk(action_chunk, now=0.0)
    command = runtime.dispatch_ready({"joint1": 0.0}, now=0.0)

    assert command is not None
    assert command.positions == [0.25]
    assert command.gripper_open_fraction == 0.25
