from __future__ import annotations

import importlib
import sys
import types


def test_openvla_worker_load_uses_local_checkpoint_loading(monkeypatch) -> None:
    fake_prismatic = types.ModuleType("prismatic")
    fake_prismatic_extern = types.ModuleType("prismatic.extern")
    fake_prismatic_hf = types.ModuleType("prismatic.extern.hf")
    fake_modeling = types.ModuleType("prismatic.extern.hf.modeling_prismatic")
    fake_processing = types.ModuleType("prismatic.extern.hf.processing_prismatic")
    fake_transformers = types.ModuleType("transformers")

    calls: dict[str, dict] = {}

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["image_processor"] = {"model_path": model_path, **kwargs}
            return object()

    class FakeProcessor:
        def __init__(self, *, image_processor, tokenizer):
            calls["processor"] = {
                "image_processor_type": type(image_processor).__name__,
                "tokenizer_type": type(tokenizer).__name__,
            }

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["tokenizer"] = {"model_path": model_path, **kwargs}
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["model"] = {"model_path": model_path, **kwargs}
            return cls()

        def to(self, device, dtype=None):
            calls["device"] = {"device": str(device), "dtype": str(dtype)}
            return self

    fake_modeling.OpenVLAForActionPrediction = FakeModel
    fake_processing.PrismaticProcessor = FakeProcessor
    fake_processing.PrismaticImageProcessor = FakeImageProcessor
    fake_transformers.AutoTokenizer = FakeTokenizer

    monkeypatch.setitem(sys.modules, "prismatic", fake_prismatic)
    monkeypatch.setitem(sys.modules, "prismatic.extern", fake_prismatic_extern)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf", fake_prismatic_hf)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.modeling_prismatic", fake_modeling)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.processing_prismatic", fake_processing)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

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

    assert calls["image_processor"]["local_files_only"] is True
    assert calls["tokenizer"]["local_files_only"] is True
    assert calls["processor"] == {"image_processor_type": "object", "tokenizer_type": "FakeTokenizer"}
    assert calls["model"]["local_files_only"] is True
    assert calls["model"]["attn_implementation"] is None
    assert calls["device"] == {"device": "cpu", "dtype": "torch.float32"}
