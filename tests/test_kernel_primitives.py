from __future__ import annotations

import numpy as np
import pytest

from roboneuron_core.kernel import (
    ActionContract,
    ExecutionSession,
    ExecutionSessionStatus,
    ExecutionTrace,
    HealthLevel,
    HealthStatus,
    RuntimeProfile,
    StateSnapshot,
)


def test_state_snapshot_round_trip_vector() -> None:
    snapshot = StateSnapshot.from_vector(
        [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.2],
        frame="base",
        metadata={"source_topic": "/task_space_state"},
    )

    np.testing.assert_allclose(
        snapshot.as_vector(),
        np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0], dtype=np.float64),
    )
    assert snapshot.frame == "base"
    assert snapshot.metadata["source_topic"] == "/task_space_state"


def test_state_snapshot_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="state vector"):
        StateSnapshot.from_vector([0.1, 0.2])


def test_action_contract_validates_raw_action_chunks() -> None:
    contract = ActionContract.raw_action_chunk(
        protocol="normalized_cartesian_velocity",
        action_dim=7,
    )

    validated = contract.validate_action_matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    assert validated.shape == (1, 7)
    assert contract.transport == "raw_action_chunk"


def test_action_contract_rejects_dim_mismatch() -> None:
    contract = ActionContract.raw_action_chunk(
        protocol="normalized_cartesian_velocity",
        action_dim=7,
    )

    with pytest.raises(ValueError, match="expects action_dim=7"):
        contract.validate_action_matrix([[0.0, 0.0, 0.0]])


def test_execution_session_tracks_lifecycle_transitions() -> None:
    profile = RuntimeProfile.edge_control(
        name="fr3_real",
        deployment_mode="local",
        robot_backend="franka",
        action_transport="raw_action_chunk",
        action_protocol="normalized_cartesian_velocity",
        state_source="task_space_state",
        vendor_stack=("franka_ros2", "libfranka"),
    )
    session = ExecutionSession.create(
        owner="roboneuron-control",
        action_contract=ActionContract.raw_action_chunk(
            protocol="normalized_cartesian_velocity",
            action_dim=7,
        ),
        runtime_profile=profile,
        now=1.0,
    )

    session.mark_starting(now=2.0, details={"raw_action_topic": "/raw_action_chunk"})
    session.mark_running(now=3.0, details={"pid": 4321})
    session.mark_stopped(now=4.0, reason="requested stop", details={"requested_by": "test"})

    assert session.status == ExecutionSessionStatus.STOPPED
    assert session.started_at_sec == 3.0
    assert session.finished_at_sec == 4.0
    assert isinstance(session.trace, ExecutionTrace)
    assert session.trace is not None
    assert session.trace.profile_name == "fr3_real"
    assert session.trace.runtime_layer == "edge"
    assert session.trace.action_transport == "raw_action_chunk"
    assert [event.status for event in session.history] == [
        ExecutionSessionStatus.CREATED,
        ExecutionSessionStatus.STARTING,
        ExecutionSessionStatus.RUNNING,
        ExecutionSessionStatus.STOPPED,
    ]
    assert session.history[0].details["profile_name"] == "fr3_real"
    assert session.history[1].details["raw_action_topic"] == "/raw_action_chunk"
    assert session.history[2].details["pid"] == 4321
    assert session.history[3].details["requested_by"] == "test"


def test_execution_session_tracks_failure() -> None:
    session = ExecutionSession.create(owner="roboneuron-vla", now=10.0)

    session.mark_failed("runtime crashed", now=12.0)

    assert session.status == ExecutionSessionStatus.FAILED
    assert session.failure_reason == "runtime crashed"
    assert session.finished_at_sec == 12.0


def test_health_status_helpers_cover_idle_ready_and_error() -> None:
    idle = HealthStatus.idle("roboneuron-control", checked_at_sec=1.0)
    ready = HealthStatus.ready("roboneuron-vla", checked_at_sec=2.0)
    error = HealthStatus.error("roboneuron-vla", summary="load failed", checked_at_sec=3.0)

    assert idle.level == HealthLevel.IDLE
    assert idle.is_healthy
    assert ready.level == HealthLevel.READY
    assert ready.is_healthy
    assert error.level == HealthLevel.ERROR
    assert not error.is_healthy


def test_runtime_profile_helpers_capture_core_and_edge_roles() -> None:
    control_profile = RuntimeProfile.edge_control(
        name="fr3_real",
        deployment_mode="local",
        robot_backend="franka",
        action_transport="raw_action_chunk",
        action_protocol="normalized_cartesian_velocity",
        state_source="task_space_state",
        vendor_stack=("franka_ros2", "libfranka"),
    )
    vla_profile = RuntimeProfile.core_vla(
        name="openvla-oft",
        deployment_mode="local",
        model_runtime="openvla-oft",
        action_transport="raw_action_chunk",
        action_protocol="normalized_cartesian_velocity",
        state_source="task_space_state",
    )

    assert control_profile.layer == "edge"
    assert control_profile.robot_backend == "franka"
    assert control_profile.vendor_stack == ("franka_ros2", "libfranka")
    assert vla_profile.layer == "core"
    assert vla_profile.model_runtime == "openvla-oft"
