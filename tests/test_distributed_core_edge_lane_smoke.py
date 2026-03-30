from __future__ import annotations

import sys
import types

import numpy as np
from PIL import Image

from roboneuron_core.kernel import ActuationCommand
from roboneuron_core.servers.vla_server import (
    EEF_DELTA_CMD_TOPIC,
    _build_model_observation,
    _resolve_output_contract,
    _resolve_output_topic,
)
from roboneuron_core.utils.raw_action_chunk import (
    RAW_ACTION_CHUNK_TOPIC,
    array_to_raw_action_chunk_message,
    raw_action_chunk_message_to_action_chunk,
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

    fake_interfaces_msg.RawActionChunk = FakeRawActionChunk
    fake_interfaces.msg = fake_interfaces_msg

    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_interfaces_msg)


class _RecordingResolver:
    def __init__(self) -> None:
        self.intent_frames: list[str] = []
        self.intent_gripper_targets: list[float | None] = []

    def resolve(self, intent, joint_positions):  # noqa: ANN001
        del joint_positions
        self.intent_frames.append(intent.frame)
        self.intent_gripper_targets.append(intent.gripper_open_fraction)
        return ActuationCommand(
            joint_names=["joint1"],
            positions=[float(intent.gripper_open_fraction or 0.0)],
            gripper_open_fraction=intent.gripper_open_fraction,
        )


def test_openvla_oft_raw_action_chunk_contract_drives_edge_runtime(monkeypatch) -> None:
    _install_fake_interface_messages(monkeypatch)

    primary = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    wrist = Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8))
    state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.8], dtype=np.float32)

    observation, kwargs = _build_model_observation(
        model_name="openvla-oft",
        primary_image=primary,
        instruction="reach the target handle",
        wrist_image=wrist,
        task_space_state=state,
    )

    assert kwargs == {}
    np.testing.assert_allclose(observation["state"], state)
    assert observation["instruction"] == "reach the target handle"

    output_mode, protocol, frame = _resolve_output_contract(
        "openvla-oft",
        "auto",
        None,
        "tool",
    )
    output_topic = _resolve_output_topic(EEF_DELTA_CMD_TOPIC, output_mode)

    assert output_mode == "raw_action_chunk"
    assert output_topic == RAW_ACTION_CHUNK_TOPIC

    model_action = np.array(
        [
            [0.2, -0.4, 0.0, 0.1, 0.0, -0.2, 0.75],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
        ],
        dtype=np.float64,
    )
    action_msg = array_to_raw_action_chunk_message(
        model_action,
        protocol=protocol,
        frame=frame,
        step_duration_sec=0.05,
    )
    action_chunk = raw_action_chunk_message_to_action_chunk(action_msg)

    resolver = _RecordingResolver()
    runtime = ControlRuntime(resolver)
    runtime.queue_action_chunk(action_chunk, now=0.0)

    first_command = runtime.dispatch_ready({"joint1": 0.0}, now=0.0)
    assert first_command is not None
    assert first_command.positions == [0.75]
    assert first_command.gripper_open_fraction == 0.75
    assert resolver.intent_frames == ["tool"]

    assert runtime.dispatch_ready({"joint1": 0.0}, now=0.04) is None

    second_command = runtime.dispatch_ready({"joint1": 0.0}, now=0.05)
    assert second_command is not None
    assert second_command.positions == [0.25]
    assert second_command.gripper_open_fraction == 0.25
    assert resolver.intent_frames == ["tool", "tool"]
    assert resolver.intent_gripper_targets == [0.75, 0.25]
