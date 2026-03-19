"""Tests for dummy adapters used in pipeline validation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from roboneuron_core.adapters.camera import get_registry as get_camera_registry
from roboneuron_core.adapters.camera.dummy_camera import DummyCameraWrapper
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


def test_dummy_vla_wrapper_predicts_action() -> None:
    wrapper = DummyVLAWrapper(img_size=16, action_dim=7)
    wrapper.load()

    image = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    action = wrapper.predict_action(image=image, instruction="do task")

    assert action.shape == (1, 7)
    np.testing.assert_allclose(
        action,
        wrapper.predict_action(image=image, instruction="do task"),
    )
def test_dummy_registries_expose_expected_keys() -> None:
    assert "dummy" in get_camera_registry()
    assert "dummy" in get_vla_registry()
