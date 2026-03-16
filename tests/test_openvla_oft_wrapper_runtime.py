from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from roboneuron_core.adapters.vla.openvla_oft import OpenVLAOFTWrapper


class FakeRuntime:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.loaded = False
        self.closed = False
        self.calls: list[dict] = []

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def predict_action(self, *, observation, instruction, unnorm_key, extra_predict_kwargs):
        self.calls.append(
            {
                "observation_keys": sorted(observation.keys()),
                "instruction": instruction,
                "unnorm_key": unnorm_key,
                "state": observation.get("state"),
                "extra_predict_kwargs": dict(extra_predict_kwargs or {}),
            }
        )
        return np.zeros((1, 7), dtype=np.float32)


def test_openvla_oft_wrapper_uses_subprocess_runtime(monkeypatch) -> None:
    fake_runtime = FakeRuntime()
    monkeypatch.setattr(
        "roboneuron_core.adapters.vla.openvla_oft.OpenVLAOFTSubprocessClient",
        lambda **kwargs: fake_runtime,
    )

    wrapper = OpenVLAOFTWrapper(
        "checkpoints/openvla-oft/openvla-oft-pick-banana",
        runtime_python="/tmp/openvla-oft/bin/python",
        attn_implementation=None,
        dtype=torch.float32,
        default_unnorm_key="vr_banana",
    )
    wrapper.load()

    action = wrapper.predict_action(
        image=Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)),
        instruction="pick banana",
        proprio=np.zeros((7,), dtype=np.float32),
    )

    assert fake_runtime.loaded
    np.testing.assert_allclose(action, np.zeros((1, 7), dtype=np.float32))
    assert len(fake_runtime.calls) == 1
    assert fake_runtime.calls[0]["observation_keys"] == ["full_image", "instruction", "state"]
    assert fake_runtime.calls[0]["instruction"] == "pick banana"
    assert fake_runtime.calls[0]["unnorm_key"] == "vr_banana"
    np.testing.assert_allclose(fake_runtime.calls[0]["state"], np.zeros((7,), dtype=np.float32))
    assert fake_runtime.calls[0]["extra_predict_kwargs"] == {}

    wrapper.close()
    assert fake_runtime.closed
