from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from roboneuron_core.adapters.vla.openvla import OpenVLAWrapper


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

    def predict_action(self, *, image, instruction, unnorm_key, extra_predict_kwargs):
        self.calls.append(
            {
                "size": image.size,
                "instruction": instruction,
                "unnorm_key": unnorm_key,
                "extra_predict_kwargs": dict(extra_predict_kwargs or {}),
            }
        )
        return np.zeros((1, 7), dtype=np.float32)


def test_openvla_wrapper_uses_subprocess_runtime(monkeypatch) -> None:
    fake_runtime = FakeRuntime()
    monkeypatch.setattr(
        "roboneuron_core.adapters.vla.openvla.OpenVLASubprocessClient",
        lambda **kwargs: fake_runtime,
    )

    wrapper = OpenVLAWrapper(
        "checkpoints/openvla/openvla-7b",
        runtime_python="/tmp/openvla/bin/python",
        attn_implementation=None,
        dtype=torch.float32,
        default_unnorm_key="bridge_orig",
    )
    wrapper.load()

    action = wrapper.predict_action(
        image=Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)),
        instruction="pick up the object",
    )

    assert fake_runtime.loaded
    np.testing.assert_allclose(action, np.zeros((1, 7), dtype=np.float32))
    assert fake_runtime.calls == [
        {
            "size": (16, 16),
            "instruction": "pick up the object",
            "unnorm_key": "bridge_orig",
            "extra_predict_kwargs": {},
        }
    ]

    wrapper.close()
    assert fake_runtime.closed
