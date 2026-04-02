from __future__ import annotations

from pathlib import Path

from roboneuron_core.kernel import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    ActionChunk,
    ActuationCommand,
    MotionIntent,
    RawActionStep,
)
from roboneuron_edge.runtime.control_runtime import ControlRuntime, URDFKinematicsResolver


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


def test_control_runtime_resolves_motion_intent_directly() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver)

    intent = MotionIntent(
        mode="cartesian_delta",
        arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        gripper_open_fraction=0.6,
        frame="tool",
    )

    command = runtime.resolve_intent(intent, {"joint1": 0.0})

    assert command.positions == [0.6]
    assert resolver.calls[0][0] == intent


def test_control_runtime_can_queue_motion_intents_directly() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver)

    runtime.queue_intents(
        (
            MotionIntent(
                mode="cartesian_delta",
                arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                gripper_open_fraction=0.2,
                frame="tool",
            ),
            MotionIntent(
                mode="cartesian_delta",
                arm=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                gripper_open_fraction=0.8,
                frame="tool",
            ),
        ),
        step_duration_sec=0.05,
        now=0.0,
    )

    first = runtime.dispatch_ready({"joint1": 0.0}, now=0.0)
    second = runtime.dispatch_ready({"joint1": 0.0}, now=0.05)

    assert first is not None
    assert first.gripper_open_fraction == 0.2
    assert second is not None
    assert second.gripper_open_fraction == 0.8


def test_control_runtime_resamples_chunk_steps_for_local_dispatch() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver, raw_action_dispatch_period_sec=0.02)

    chunk = ActionChunk(
        steps=(
            RawActionStep(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.10,
    )

    runtime.queue_action_chunk(chunk, now=0.0)
    outputs = [
        runtime.dispatch_ready({"joint1": 0.0}, now=dispatch_at)
        for dispatch_at in (0.0, 0.02, 0.04, 0.06, 0.08)
    ]

    assert all(output is not None for output in outputs)
    assert runtime.scheduler.step_duration_sec == 0.02
    assert len(resolver.calls) == 5
    assert resolver.calls[0][0].metadata["edge_substeps"] == 5
    assert resolver.calls[0][0].metadata["edge_substep_index"] == 0
    assert resolver.calls[-1][0].metadata["edge_substep_index"] == 4
    assert resolver.calls[0][0].arm[0] == 0.015
    assert resolver.calls[-1][0].arm[0] == 0.015


def test_control_runtime_can_dispatch_intents_without_resolving() -> None:
    resolver = FakeResolver()
    runtime = ControlRuntime(resolver, raw_action_dispatch_period_sec=0.02)

    chunk = ActionChunk(
        steps=(
            RawActionStep(
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
            ),
        ),
        step_duration_sec=0.10,
    )

    runtime.queue_action_chunk(chunk, now=0.0)

    first = runtime.dispatch_ready_intent(now=0.0)
    second = runtime.dispatch_ready_intent(now=0.02)

    assert first is not None
    assert second is not None
    assert first.arm[0] == 0.015
    assert second.arm[0] == 0.015
    assert resolver.calls == []


def _fr3_test_joint_positions() -> dict[str, float]:
    return {
        "fr3_joint1": 0.0,
        "fr3_joint2": 0.0,
        "fr3_joint3": 0.0,
        "fr3_joint4": -1.59695,
        "fr3_joint5": 0.0,
        "fr3_joint6": 2.5307,
        "fr3_joint7": 0.0,
        "fr3_finger_joint1": 0.0,
        "fr3_finger_joint2": 0.0,
    }


def test_urdf_kinematics_resolver_limits_joint_delta_per_step() -> None:
    resolver = URDFKinematicsResolver(str(Path(__file__).resolve().parents[1] / "urdf" / "fr3.urdf"))
    joint_positions = _fr3_test_joint_positions()

    command = resolver.resolve(
        MotionIntent(
            mode="cartesian_delta",
            arm=[0.075, 0.0, 0.0, 0.0, 0.0, 0.0],
            gripper_open_fraction=0.5,
            frame="tool",
        ),
        joint_positions,
    )

    arm_positions = dict(zip(command.joint_names, command.positions, strict=False))
    arm_deltas = [
        arm_positions[name] - joint_positions[name]
        for name in resolver.active_joint_names
    ]

    assert len(resolver.active_joint_names) == 7
    assert max(abs(delta) for delta in arm_deltas) <= 0.2000001
    assert command.gripper_open_fraction == 0.5


def test_urdf_kinematics_resolver_accepts_base_frame_delta() -> None:
    resolver = URDFKinematicsResolver(str(Path(__file__).resolve().parents[1] / "urdf" / "fr3.urdf"))
    joint_positions = _fr3_test_joint_positions()

    command = resolver.resolve(
        MotionIntent(
            mode="cartesian_delta",
            arm=[0.0, 0.0, 0.01, 0.0, 0.0, 0.02],
            gripper_open_fraction=0.0,
            frame="base",
        ),
        joint_positions,
    )

    assert len(command.joint_names) == 7
    assert all(position == position for position in command.positions)
