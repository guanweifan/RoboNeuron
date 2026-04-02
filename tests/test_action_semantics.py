from __future__ import annotations

import numpy as np
import pytest

from roboneuron_core.kernel import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    ActionChunk,
    NormalizedCartesianVelocityConfig,
    RawActionStep,
    motion_intent_from_eef_delta,
    motion_intent_from_raw_step,
    motion_intents_from_action_chunk,
)


def test_motion_intent_from_eef_delta_preserves_cartesian_delta() -> None:
    intent = motion_intent_from_eef_delta([0.01, -0.02, 0.03, 0.1, -0.2, 0.3, 0.75])

    np.testing.assert_allclose(intent.arm, np.array([0.01, -0.02, 0.03, 0.1, -0.2, 0.3]))
    assert intent.gripper_open_fraction == 0.75
    assert intent.mode == "cartesian_delta"
    assert intent.frame == "tool"


def test_normalized_velocity_protocol_scales_and_inverts_gripper() -> None:
    step = RawActionStep(
        [1.2, -0.5, 0.25, 0.5, -2.0, 0.2, 0.8],
        protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
        frame="base",
    )

    intent = motion_intent_from_raw_step(
        step,
        normalized_velocity_config=NormalizedCartesianVelocityConfig(
            max_linear_delta=0.075,
            max_rotation_delta=0.15,
            invert_gripper=True,
        ),
    )

    np.testing.assert_allclose(
        intent.arm,
        np.array([0.075, -0.0375, 0.01875, 0.075, -0.15, 0.03]),
    )
    assert intent.gripper_open_fraction == pytest.approx(0.2)
    assert intent.frame == "base"


def test_motion_intents_from_action_chunk_interprets_all_steps() -> None:
    chunk = ActionChunk(
        steps=(
            RawActionStep(
                [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                protocol="eef_delta",
                frame="tool",
            ),
            RawActionStep(
                [1.0, -1.0, 0.5, 0.2, -0.2, 0.4, 0.25],
                protocol=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
                frame="base",
            ),
        ),
        step_duration_sec=0.2,
    )

    intents = motion_intents_from_action_chunk(
        chunk,
        normalized_velocity_config=NormalizedCartesianVelocityConfig(
            max_linear_delta=0.1,
            max_rotation_delta=0.2,
        ),
    )

    assert len(intents) == 2
    assert intents[0].frame == "tool"
    assert intents[0].gripper_open_fraction == pytest.approx(0.9)
    np.testing.assert_allclose(intents[0].arm, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert intents[1].frame == "base"
    assert intents[1].gripper_open_fraction == pytest.approx(0.25)
    np.testing.assert_allclose(intents[1].arm, np.array([0.1, -0.1, 0.05, 0.04, -0.04, 0.08]))
