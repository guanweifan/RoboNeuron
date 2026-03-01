"""Contract tests for the CALVIN adapter wrapper."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from roboneuron_core.adapters.robot.calvin_adapter import (
    CalvinAdapterWrapper,
)


class FakeEnv:
    """Minimal CALVIN-like env for adapter contract tests."""

    def __init__(self) -> None:
        self._obs = {
            "rgb_obs": {
                "rgb_static": np.full((2, 2, 3), 1, dtype=np.uint8),
                "rgb_gripper": np.full((2, 2, 3), 2, dtype=np.uint8),
            },
            "robot_obs": np.arange(15, dtype=np.float32),
        }
        self.step_calls = 0
        self.last_action: np.ndarray | None = None

    def reset(self) -> dict[str, Any]:
        return self._obs

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self.last_action = action
        self.step_calls += 1
        reward = 0.5
        done = self.step_calls >= 2
        info = {"step_calls": self.step_calls}
        return self._obs, reward, done, info


class TestCalvinAdapter(unittest.TestCase):
    def _build_wrapper(self, step_substeps: int = 1) -> Any:
        return CalvinAdapterWrapper(
            dataset_path="/tmp/fake",
            task_id=0,
            show_gui=False,
            visual_resolution=200,
            step_substeps=step_substeps,
            auto_init_env=False,
        )

    def test_observation_mapping_and_instruction(self) -> None:
        wrapper: Any = self._build_wrapper()
        wrapper.env = FakeEnv()
        wrapper.raw_obs = wrapper.env.reset()
        wrapper.task_sequence = [{"language": "do something"}]
        wrapper.current_step_idx = 0
        wrapper._update_current_instruction()

        obs = wrapper.obtain_observation()

        self.assertIn("agentview_image", obs["visual_observation"])
        self.assertIn("eye_in_hand_image", obs["visual_observation"])
        np.testing.assert_array_equal(
            obs["visual_observation"]["agentview_image"], wrapper.raw_obs["rgb_obs"]["rgb_static"]
        )
        np.testing.assert_array_equal(
            obs["visual_observation"]["eye_in_hand_image"], wrapper.raw_obs["rgb_obs"]["rgb_gripper"]
        )

        self.assertEqual(obs["robot_state"].shape, (8,))
        expected_state = np.concatenate(
            [wrapper.raw_obs["robot_obs"][7:14], wrapper.raw_obs["robot_obs"][6:7]]
        )
        np.testing.assert_array_equal(obs["robot_state"], expected_state)
        self.assertEqual(obs["instruction"], "do something")

    def test_step_scales_gripper_and_accumulates_substeps(self) -> None:
        wrapper: Any = self._build_wrapper(step_substeps=2)
        wrapper.env = FakeEnv()
        wrapper.raw_obs = wrapper.env.reset()
        wrapper.task_sequence = [{"language": "task"}]
        wrapper.current_step_idx = 0
        wrapper._update_current_instruction()

        action = np.zeros(7, dtype=np.float32)
        action[-1] = 0.5
        result = wrapper.step(action)

        self.assertIsNotNone(wrapper.env.last_action)
        assert wrapper.env.last_action is not None
        self.assertAlmostEqual(wrapper.env.last_action[-1], 1.0, places=5)
        self.assertEqual(wrapper.env.step_calls, 2)
        self.assertAlmostEqual(result["reward"], 1.0)
        self.assertTrue(result["done"])
        self.assertIn("step_calls", result["task_info"])

    def test_obtain_observation_raises_when_env_missing(self) -> None:
        wrapper: Any = self._build_wrapper()
        wrapper.env = None
        wrapper.raw_obs = {"robot_obs": np.zeros(15, dtype=np.float32)}
        with self.assertRaises(RuntimeError):
            wrapper.obtain_observation()

    def test_obtain_observation_raises_when_raw_obs_missing(self) -> None:
        wrapper: Any = self._build_wrapper()
        wrapper.env = FakeEnv()
        wrapper.raw_obs = None
        with self.assertRaises(RuntimeError):
            wrapper.obtain_observation()


if __name__ == "__main__":
    unittest.main()
