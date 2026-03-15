from __future__ import annotations

import importlib
import sys

import numpy as np


class FakeWrapper:
    instances: list[FakeWrapper] = []

    def __init__(self, model_path: str, **kwargs) -> None:
        self.model_path = model_path
        self.kwargs = kwargs
        self.loaded = False
        self.closed = False
        self.calls: list[dict] = []
        type(self).instances.append(self)

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def predict_action(self, *, image, instruction, **kwargs):
        self.calls.append(
            {
                "size": image.size,
                "instruction": instruction,
                "kwargs": dict(kwargs),
            }
        )
        return np.zeros((1, 7), dtype=np.float32)


def test_run_local_test_without_ros(monkeypatch, capsys) -> None:
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    FakeWrapper.instances.clear()
    monkeypatch.setattr(vla_server, "get_registry", lambda: {"openvla": FakeWrapper})
    monkeypatch.setattr(
        vla_server,
        "_resolve_model_spec",
        lambda model_name, model_path: (
            "checkpoints/openvla/openvla-7b",
            {"default_unnorm_key": "bridge_orig"},
        ),
    )
    monkeypatch.setattr(
        vla_server,
        "resolve_accel_configs",
        lambda model_name, accel_method, accel_level: ({"window": 4}, {"ratio": 0.5}),
    )

    result = vla_server._run_local_test(
        model_name="openvla",
        model_path=None,
        instruction="pick up the blue bowl",
        accel_method="fastv",
        accel_level="balanced",
    )

    captured = capsys.readouterr()

    assert result == 0
    assert "Local test succeeded." in captured.out
    assert len(FakeWrapper.instances) == 1
    wrapper = FakeWrapper.instances[0]
    assert wrapper.loaded
    assert wrapper.closed
    assert wrapper.calls == [
        {
            "size": (64, 64),
            "instruction": "pick up the blue bowl",
            "kwargs": {
                "accel_method": "fastv",
                "accel_config": {"window": 4},
                "prune_config": {"ratio": 0.5},
            },
        }
    ]
