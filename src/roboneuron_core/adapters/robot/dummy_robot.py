"""Deterministic in-memory robot adapter for pipeline tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import AdapterWrapper


class DummyRobotAdapterWrapper(AdapterWrapper):
    """Small adapter that emulates robot state transitions without external dependencies."""

    def __init__(
        self,
        image_size: int = 128,
        instruction: str = "execute dummy task",
        action_scale: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.image_size = int(image_size)
        self.instruction = instruction
        self.action_scale = float(action_scale)
        self.env: DummyRobotAdapterWrapper | None = None
        self.step_count = 0
        self.robot_state = np.zeros(8, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.create_simulation_environment()

    def create_simulation_environment(self) -> None:
        self.env = self
        self.step_count = 0
        self.robot_state = np.zeros(8, dtype=np.float32)
        self.last_action = np.zeros(7, dtype=np.float32)

    def close(self) -> None:
        self.env = None

    def _render_image(self) -> np.ndarray:
        width = self.image_size
        height = self.image_size
        x_gradient = np.linspace(0, 255, width, dtype=np.float32)[None, :]
        y_gradient = np.linspace(0, 255, height, dtype=np.float32)[:, None]

        x_offset = float(self.robot_state[0] * 40.0)
        y_offset = float(self.robot_state[1] * 40.0)
        grip_offset = float((self.robot_state[6] + 1.0) * 50.0)

        red = np.clip(np.repeat(y_gradient, width, axis=1) + y_offset, 0, 255)
        green = np.clip(np.repeat(np.flip(x_gradient, axis=1), height, axis=0) + grip_offset, 0, 255)
        blue = np.clip(np.repeat(x_gradient, height, axis=0) + x_offset, 0, 255)
        return np.stack([red, green, blue], axis=2).astype(np.uint8)

    def obtain_observation(self) -> dict[str, Any]:
        image = self._render_image()
        return {
            "visual_observation": {
                "agentview_image": image,
                "rgb_static": image,
            },
            "robot_state": self.robot_state.copy(),
            "instruction": self.instruction,
            "task_info": {
                "step_count": self.step_count,
                "last_action": self.last_action.tolist(),
            },
        }

    def step(self, action: Any) -> dict[str, Any]:
        action_np = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_np.size != 7:
            raise ValueError(f"DummyRobotAdapterWrapper expects a 7D action, got shape {action_np.shape}.")

        self.step_count += 1
        self.last_action = action_np.copy()
        self.robot_state[:7] = np.clip(
            self.robot_state[:7] + action_np * self.action_scale,
            -1.0,
            1.0,
        )
        self.robot_state[7] = float(self.step_count)
        return {
            "observation": self.obtain_observation(),
            "reward": float(self.step_count),
            "done": False,
            "task_info": {
                "step_count": self.step_count,
                "last_action": self.last_action.tolist(),
            },
        }
