from __future__ import annotations

from roboneuron_core.utils.control_runtime import (
    ActionChunk,
    ActuationCommand,
    ControlRuntime,
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    RawActionStep,
)


class FakeResolver:
    def __init__(self) -> None:
        self.calls = []

    def resolve(self, intent, joint_positions):
        self.calls.append((intent, dict(joint_positions)))
        return ActuationCommand(joint_names=["joint1"], positions=[float(intent.gripper_open_fraction or 0.0)])


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
    assert not_ready is None
    assert second is not None
    assert second.positions == [0.75]
    assert len(resolver.calls) == 2
