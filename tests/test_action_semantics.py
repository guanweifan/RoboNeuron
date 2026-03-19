from __future__ import annotations

import numpy as np
import pytest

from roboneuron_core.utils.control_runtime import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    NormalizedCartesianVelocityConfig,
    RawActionStep,
    motion_intent_from_eef_delta,
    motion_intent_from_raw_step,
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
