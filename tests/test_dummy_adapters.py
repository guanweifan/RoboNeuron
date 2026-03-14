"""Tests for dummy adapters used in pipeline validation."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest

from roboneuron_core.adapters.camera import get_registry as get_camera_registry
from roboneuron_core.adapters.camera.dummy_camera import DummyCameraWrapper
from roboneuron_core.adapters.robot import get_registry as get_robot_registry
from roboneuron_core.adapters.robot.dummy_robot import DummyRobotAdapterWrapper
from roboneuron_core.adapters.vla import get_registry as get_vla_registry
from roboneuron_core.adapters.vla.dummy_vla import DummyVLAWrapper


def test_dummy_camera_wrapper_generates_deterministic_frame() -> None:
    wrapper = DummyCameraWrapper(width=8, height=4)

    ok, frame = wrapper.read()
    assert not ok
    assert frame is None

    wrapper.open()
    ok, frame = wrapper.read()
    assert ok
    assert frame is not None
    assert frame.shape == (4, 8, 3)
    assert frame.dtype == np.uint8
    assert frame[0, 0].tolist() == [0, 255, 0]
    assert wrapper.is_opened()

    wrapper.close()
    assert not wrapper.is_opened()


def test_dummy_vla_wrapper_supports_accel_and_prune_configs() -> None:
    wrapper = DummyVLAWrapper(img_size=16, action_dim=7)
    wrapper.load()

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    action = wrapper.predict_action(
        image=image,
        instruction="do task",
        accel_method="fastv",
        accel_config={"use_fastv": True, "fastv_k": 4, "fastv_r": 0.75},
        prune_config={"layer": "dummy_block", "ratio": 0.25},
    )

    assert action.shape == (1, 7)
    assert wrapper.model is not None
    assert wrapper.model.current_prune_ratio == pytest.approx(0.25)
    assert wrapper.model.last_fastv_config == {
        "use_fastv": True,
        "fastv_k": 4,
        "fastv_r": 0.75,
    }
    np.testing.assert_allclose(
        action,
        wrapper.predict_action(
            image=image,
            instruction="do task",
            accel_method="fastv",
            accel_config={"use_fastv": True, "fastv_k": 4, "fastv_r": 0.75},
            prune_config={"layer": "dummy_block", "ratio": 0.25},
        ),
    )


def test_dummy_robot_adapter_returns_standardized_observation_and_state_updates() -> None:
    wrapper = DummyRobotAdapterWrapper(image_size=16, instruction="test instruction", action_scale=0.5)

    observation = wrapper.obtain_observation()
    assert observation["instruction"] == "test instruction"
    assert observation["robot_state"].shape == (8,)
    assert observation["visual_observation"]["agentview_image"].shape == (16, 16, 3)

    action = np.array([0.2, -0.1, 0.3, 0.0, 0.5, -0.25, 1.0], dtype=np.float32)
    result = wrapper.step(action)

    np.testing.assert_allclose(wrapper.last_action, action)
    np.testing.assert_allclose(
        wrapper.robot_state[:7],
        np.array([0.1, -0.05, 0.15, 0.0, 0.25, -0.125, 0.5], dtype=np.float32),
    )
    assert wrapper.robot_state[7] == pytest.approx(1.0)
    assert result["reward"] == pytest.approx(1.0)
    assert result["task_info"]["step_count"] == 1


def test_dummy_registries_expose_expected_keys() -> None:
    assert "dummy" in get_camera_registry()
    assert "dummy" in get_robot_registry()
    assert "dummy" in get_vla_registry()
