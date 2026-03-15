from __future__ import annotations

import importlib
import sys
import types


def test_openvla_worker_load_uses_noninteractive_trust_remote_code(monkeypatch) -> None:
    fake_prismatic = types.ModuleType("prismatic")
    fake_prismatic_extern = types.ModuleType("prismatic.extern")
    fake_prismatic_hf = types.ModuleType("prismatic.extern.hf")
    fake_modeling = types.ModuleType("prismatic.extern.hf.modeling_prismatic")
    fake_processing = types.ModuleType("prismatic.extern.hf.processing_prismatic")

    calls: dict[str, dict] = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["processor"] = {"model_path": model_path, **kwargs}
            return object()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["model"] = {"model_path": model_path, **kwargs}
            return cls()

        def to(self, device):
            calls["device"] = {"device": str(device)}
            return self

    fake_modeling.OpenVLAForActionPrediction = FakeModel
    fake_processing.PrismaticProcessor = FakeProcessor

    monkeypatch.setitem(sys.modules, "prismatic", fake_prismatic)
    monkeypatch.setitem(sys.modules, "prismatic.extern", fake_prismatic_extern)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf", fake_prismatic_hf)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.modeling_prismatic", fake_modeling)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.processing_prismatic", fake_processing)

    sys.modules.pop("roboneuron_core.runtime.openvla_worker", None)
    openvla_worker = importlib.import_module("roboneuron_core.runtime.openvla_worker")

    worker = openvla_worker.OpenVLAWorker(
        model_path="checkpoints/openvla/openvla-7b",
        attn_implementation="flash_attention_2",
        dtype_name="float32",
        device_name="cpu",
        low_cpu_mem_usage=True,
    )
    worker.load()

    assert calls["processor"]["trust_remote_code"] is True
    assert calls["model"]["trust_remote_code"] is True
    assert calls["model"]["attn_implementation"] is None
    assert calls["device"]["device"] == "cpu"
