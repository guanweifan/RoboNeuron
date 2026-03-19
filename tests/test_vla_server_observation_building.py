from __future__ import annotations

import numpy as np
from PIL import Image

from roboneuron_core.servers.vla_server import _build_model_observation


def test_build_model_observation_keeps_openvla_single_image_contract() -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))

    observation, kwargs = _build_model_observation(
        model_name="openvla",
        primary_image=image,
        instruction="pick object",
        wrist_image=Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8)),
        task_space_state=np.ones((7,), dtype=np.float32),
    )

    assert observation is image
    assert kwargs == {}


def test_build_model_observation_adds_wrist_and_state_for_openvla_oft() -> None:
    primary = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    wrist = Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8))
    state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)

    observation, kwargs = _build_model_observation(
        model_name="openvla-oft",
        primary_image=primary,
        instruction="pick object",
        wrist_image=wrist,
        task_space_state=state,
    )

    assert kwargs == {}
    assert sorted(observation.keys()) == ["full_image", "instruction", "state", "wrist_image"]
    assert observation["full_image"] is primary
    assert observation["wrist_image"] is wrist
    np.testing.assert_allclose(observation["state"], state)
