from __future__ import annotations

from roboneuron_core.kernel import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    ActionChunk,
    ActuationCommand,
    RawActionStep,
)
from roboneuron_edge.runtime.control_runtime import ControlRuntime


class FakeResolver:
    def __init__(self) -> None:
        self.calls = []

    def resolve(self, intent, joint_positions):
        self.calls.append((intent, dict(joint_positions)))
        return ActuationCommand(
            joint_names=["joint1"],
            positions=[float(intent.gripper_open_fraction or 0.0)],
            gripper_open_fraction=intent.gripper_open_fraction,
        )


def test_control_runtime_dispatches_chunk_steps_over_time() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver)

    chunk = ActionChunk(
        steps=(
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.05,
    )

    runtime.queue_action_chunk(chunk, now=0.0)

    first = runtime.dispatch_ready({"joint1": 0.0}, now=0.0)
    not_ready = runtime.dispatch_ready({"joint1": 0.0}, now=0.01)
    second = runtime.dispatch_ready({"joint1": 0.0}, now=0.06)

    assert first is not None
    assert first.positions == [0.25]
    assert first.gripper_open_fraction == 0.25
    assert not_ready is None
    assert second is not None
    assert second.positions == [0.75]
    assert second.gripper_open_fraction == 0.75
    assert len(resolver.calls) == 2


def test_control_runtime_replacing_chunk_does_not_break_dispatch_rate() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver)

    original_chunk = ActionChunk(
        steps=(
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.50],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.10,
    )
    replacement_chunk = ActionChunk(
        steps=(
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.90],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.10,
    )

    runtime.queue_action_chunk(original_chunk, now=0.0)
    first = runtime.dispatch_ready({"joint1": 0.0}, now=0.0)
    runtime.queue_action_chunk(replacement_chunk, now=0.02)

    still_waiting = runtime.dispatch_ready({"joint1": 0.0}, now=0.02)
    replacement = runtime.dispatch_ready({"joint1": 0.0}, now=0.11)

    assert first is not None
    assert first.positions == [0.25]
    assert still_waiting is None
    assert replacement is not None
    assert replacement.positions == [0.90]
    assert replacement.gripper_open_fraction == 0.90
    assert len(resolver.calls) == 2


def test_control_runtime_can_clear_pending_chunk() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver)

    chunk = ActionChunk(
        steps=(
            RawActionStep(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.10,
    )

    runtime.queue_action_chunk(chunk, now=0.0)
    runtime.clear_action_chunk()

    assert runtime.scheduler.pending_count == 0
    assert runtime.dispatch_ready({"joint1": 0.0}, now=0.0) is None
    assert resolver.calls == []
