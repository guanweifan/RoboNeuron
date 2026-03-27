from __future__ import annotations

import sys
import types

import numpy as np


def _install_fake_raw_action_chunk_module(monkeypatch) -> None:
    fake_roboneuron_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_roboneuron_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")

    class FakeRawActionChunk:
        def __init__(self) -> None:
            self.protocol = ""
            self.frame = ""
            self.action_dim = 0
            self.chunk_length = 0
            self.step_duration_sec = 0.0
            self.values = []

    fake_roboneuron_interfaces_msg.RawActionChunk = FakeRawActionChunk
    fake_roboneuron_interfaces.msg = fake_roboneuron_interfaces_msg

    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_roboneuron_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_roboneuron_interfaces_msg)


def test_raw_action_chunk_round_trip(monkeypatch) -> None:
    _install_fake_raw_action_chunk_module(monkeypatch)

    from roboneuron_core.utils.raw_action_chunk import (
        array_to_raw_action_chunk_message,
        raw_action_chunk_message_to_action_chunk,
    )

    action_chunk = np.array(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
        dtype=np.float64,
    )
    message = array_to_raw_action_chunk_message(
        action_chunk,
        protocol="normalized_cartesian_velocity",
        frame="tool",
        step_duration_sec=0.1,
    )

    decoded = raw_action_chunk_message_to_action_chunk(message)

    assert message.action_dim == 7
    assert message.chunk_length == 2
    assert message.protocol == "normalized_cartesian_velocity"
    assert message.frame == "tool"
    np.testing.assert_allclose(decoded.steps[0].values, action_chunk[0])
    np.testing.assert_allclose(decoded.steps[1].values, action_chunk[1])
    assert decoded.step_duration_sec == 0.1
