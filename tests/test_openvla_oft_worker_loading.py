from __future__ import annotations

import importlib
import json
import sys
import types


def test_openvla_oft_worker_load_infers_bridge_profile(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "openvla-oft"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "action_head--60000_checkpoint.pt").write_text("", encoding="utf-8")
    (checkpoint_dir / "proprio_projector--60000_checkpoint.pt").write_text("", encoding="utf-8")
    (checkpoint_dir / "vision_backbone--60000_checkpoint.pt").write_text("", encoding="utf-8")
    (checkpoint_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "vr_banana": {
                    "action": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                }
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_processing = types.ModuleType("prismatic.extern.hf.processing_prismatic")
    fake_modeling = types.ModuleType("prismatic.extern.hf.modeling_prismatic")
    fake_transformers = types.ModuleType("transformers")

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["image_processor"] = {"model_path": str(model_path), **kwargs}
            instance = cls()
            instance.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])
            return instance

    class FakeProcessor:
        def __init__(self, *, image_processor, tokenizer):
            calls["processor"] = {
                "image_processor_type": type(image_processor).__name__,
                "tokenizer_type": type(tokenizer).__name__,
            }
            self.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["tokenizer"] = {"model_path": str(model_path), **kwargs}
            return cls()

    class FakeVisionBackbone:
        def set_num_images_in_input(self, value):
            calls.setdefault("num_images", []).append(value)

    class FakeModel:
        def __init__(self) -> None:
            self.llm_dim = 128
            self.vision_backbone = FakeVisionBackbone()
            self.norm_stats = {
                "vr_banana": {
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7}
                }
            }

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["model"] = {"model_path": str(model_path), **kwargs}
            return cls()

        def eval(self):
            calls["eval"] = True
            return self

        def to(self, device, dtype=None):
            calls["device"] = {"device": str(device), "dtype": str(dtype)}
            return self

    fake_processing.PrismaticImageProcessor = FakeImageProcessor
    fake_processing.PrismaticProcessor = FakeProcessor
    fake_modeling.OpenVLAForActionPrediction = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer

    fake_action_heads = types.ModuleType("prismatic.models.action_heads")

    class FakeActionHead:
        def __init__(self, input_dim, hidden_dim, **kwargs):
            calls["action_head_init"] = {"input_dim": input_dim, "hidden_dim": hidden_dim, **kwargs}

        def load_state_dict(self, state_dict):
            calls["action_head_state_dict"] = state_dict

        def eval(self):
            calls["action_head_eval"] = True
            return self

        def to(self, device, dtype=None):
            calls["action_head_to"] = {"device": str(device), "dtype": str(dtype)}
            return self

    class FakeDiffusionActionHead(FakeActionHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.noise_scheduler = types.SimpleNamespace(set_timesteps=lambda value: calls.setdefault("timesteps", value))

    fake_action_heads.L1RegressionActionHead = FakeActionHead
    fake_action_heads.DiffusionActionHead = FakeDiffusionActionHead

    fake_projectors = types.ModuleType("prismatic.models.projectors")

    class FakeProprioProjector:
        def __init__(self, llm_dim, proprio_dim):
            calls["proprio_init"] = {"llm_dim": llm_dim, "proprio_dim": proprio_dim}

        def load_state_dict(self, state_dict):
            calls["proprio_state_dict"] = state_dict

        def eval(self):
            calls["proprio_eval"] = True
            return self

        def to(self, device, dtype=None):
            calls["proprio_to"] = {"device": str(device), "dtype": str(dtype)}
            return self

    class FakeNoisyActionProjector:
        def __init__(self, llm_dim):
            calls["noisy_init"] = {"llm_dim": llm_dim}

        def load_state_dict(self, state_dict):
            calls["noisy_state_dict"] = state_dict

        def eval(self):
            return self

        def to(self, device, dtype=None):
            return self

    fake_projectors.ProprioProjector = FakeProprioProjector
    fake_projectors.NoisyActionProjector = FakeNoisyActionProjector

    fake_film = types.ModuleType("prismatic.models.film_vit_wrapper")

    class FakeFiLMedPrismaticVisionBackbone:
        def __init__(self, vision_backbone, llm_dim):
            calls["film_init"] = {"llm_dim": llm_dim, "vision_backbone_type": type(vision_backbone).__name__}
            self._vision_backbone = vision_backbone

        def load_state_dict(self, state_dict):
            calls["film_state_dict"] = state_dict

        def set_num_images_in_input(self, value):
            calls.setdefault("film_num_images", []).append(value)

    fake_film.FiLMedPrismaticVisionBackbone = FakeFiLMedPrismaticVisionBackbone

    monkeypatch.setitem(sys.modules, "prismatic.models.action_heads", fake_action_heads)
    monkeypatch.setitem(sys.modules, "prismatic.models.projectors", fake_projectors)
    monkeypatch.setitem(sys.modules, "prismatic.models.film_vit_wrapper", fake_film)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.processing_prismatic", fake_processing)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.modeling_prismatic", fake_modeling)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    sys.modules.pop("roboneuron_core.runtime.openvla_oft_worker", None)
    openvla_oft_worker = importlib.import_module("roboneuron_core.runtime.openvla_oft_worker")

    monkeypatch.setattr(
        openvla_oft_worker,
        "_load_component_state_dict",
        lambda path: {"loaded_from": path.name},
    )

    worker = openvla_oft_worker.OpenVLAOFTWorker(
        model_path=str(checkpoint_dir),
        attn_implementation="flash_attention_2",
        dtype_name="float32",
        device_name="cpu",
        runtime_quantization="none",
        low_cpu_mem_usage=True,
        use_l1_regression=None,
        use_diffusion=None,
        use_film=None,
        use_proprio=None,
        num_images_in_input=None,
        num_diffusion_steps_inference=50,
        lora_rank=32,
        center_crop=True,
        unnorm_key=None,
        robot_platform=None,
        default_proprio=None,
        base_model_path=None,
    )
    worker.load()

    assert worker.robot_platform == "bridge"
    assert worker.default_unnorm_key == "vr_banana"
    assert worker.resolved_num_images_in_input == 1
    assert worker.resolved_use_proprio is True
    assert worker.resolved_use_film is True
    assert worker.resolved_use_l1_regression is True
    assert worker.resolved_use_diffusion is False
    assert calls["image_processor"] == {"model_path": str(checkpoint_dir.resolve()), "local_files_only": True}
    assert calls["tokenizer"] == {"model_path": str(checkpoint_dir.resolve()), "local_files_only": True}
    assert calls["processor"] == {"image_processor_type": "FakeImageProcessor", "tokenizer_type": "FakeTokenizer"}
    assert calls["model"]["attn_implementation"] is None
    assert calls["model"]["local_files_only"] is True
    assert calls["device"] == {"device": "cpu", "dtype": "torch.float32"}
    assert calls["proprio_init"] == {"llm_dim": 128, "proprio_dim": 7}
    assert calls["film_init"] == {"llm_dim": 128, "vision_backbone_type": "FakeVisionBackbone"}


def test_openvla_oft_worker_load_supports_4bit_quantization(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "openvla-oft"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "vr_banana": {
                    "action": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                }
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_processing = types.ModuleType("prismatic.extern.hf.processing_prismatic")
    fake_modeling = types.ModuleType("prismatic.extern.hf.modeling_prismatic")
    fake_transformers = types.ModuleType("transformers")

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            instance = cls()
            instance.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])
            calls["image_processor"] = {"model_path": str(model_path), **kwargs}
            return instance

    class FakeProcessor:
        def __init__(self, *, image_processor, tokenizer):
            del tokenizer
            self.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])
            calls["processor"] = {"image_processor_type": type(image_processor).__name__}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["tokenizer"] = {"model_path": str(model_path), **kwargs}
            return cls()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeVisionBackbone:
        def set_num_images_in_input(self, value):
            calls.setdefault("num_images", []).append(value)

    class FakeModel:
        def __init__(self) -> None:
            self.llm_dim = 128
            self.vision_backbone = FakeVisionBackbone()
            self.norm_stats = {
                "vr_banana": {
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7}
                }
            }

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            calls["model"] = {"model_path": str(model_path), **kwargs}
            return cls()

        def eval(self):
            return self

        def to(self, device, dtype=None):
            calls["device"] = {"device": str(device), "dtype": str(dtype)}
            return self

    fake_processing.PrismaticImageProcessor = FakeImageProcessor
    fake_processing.PrismaticProcessor = FakeProcessor
    fake_modeling.OpenVLAForActionPrediction = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig

    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.processing_prismatic", fake_processing)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.modeling_prismatic", fake_modeling)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    sys.modules.pop("roboneuron_core.runtime.openvla_oft_worker", None)
    openvla_oft_worker = importlib.import_module("roboneuron_core.runtime.openvla_oft_worker")

    monkeypatch.setattr(openvla_oft_worker, "_prepare_prismatic_shallow_packages", lambda: None)
    monkeypatch.setattr(
        openvla_oft_worker.OpenVLAOFTWorker,
        "_import_runtime_modules",
        lambda self: {
            "OpenVLAForActionPrediction": FakeModel,
            "PrismaticImageProcessor": FakeImageProcessor,
            "PrismaticProcessor": FakeProcessor,
            "L1RegressionActionHead": object,
            "DiffusionActionHead": object,
            "NoisyActionProjector": object,
            "ProprioProjector": object,
            "FiLMedPrismaticVisionBackbone": object,
        },
    )

    worker = openvla_oft_worker.OpenVLAOFTWorker(
        model_path=str(checkpoint_dir),
        attn_implementation="flash_attention_2",
        dtype_name="float16",
        device_name="cuda:0",
        runtime_quantization="4bit",
        low_cpu_mem_usage=True,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        use_proprio=False,
        num_images_in_input=1,
        num_diffusion_steps_inference=50,
        lora_rank=32,
        center_crop=True,
        unnorm_key="vr_banana",
        robot_platform="bridge",
        default_proprio=None,
        base_model_path=None,
    )
    worker.load()

    quantization_config = calls["model"]["quantization_config"]
    assert isinstance(quantization_config, FakeBitsAndBytesConfig)
    assert quantization_config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": openvla_oft_worker.torch.float16,
        "bnb_4bit_quant_type": "nf4",
    }
    assert calls["model"]["device_map"] == {"": 0}
    assert "device" not in calls


def test_openvla_oft_worker_quantized_load_enables_force_hooks(monkeypatch, tmp_path) -> None:
    checkpoint_dir = tmp_path / "openvla-oft"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (checkpoint_dir / "dataset_statistics.json").write_text(
        json.dumps(
            {
                "vr_banana": {
                    "action": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7},
                }
            }
        ),
        encoding="utf-8",
    )

    calls: dict[str, object] = {}

    fake_processing = types.ModuleType("prismatic.extern.hf.processing_prismatic")
    fake_modeling = types.ModuleType("prismatic.extern.hf.modeling_prismatic")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__path__ = []  # type: ignore[attr-defined]
    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")

    class FakeImageProcessor:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            instance = cls()
            instance.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])
            return instance

    class FakeProcessor:
        def __init__(self, *, image_processor, tokenizer):
            del image_processor, tokenizer
            self.image_processor = types.SimpleNamespace(input_sizes=[(3, 224, 224)])

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            return cls()

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeVisionBackbone:
        def set_num_images_in_input(self, value):
            del value

    class FakeModel:
        def __init__(self) -> None:
            self.llm_dim = 128
            self.vision_backbone = FakeVisionBackbone()
            self.norm_stats = {
                "vr_banana": {
                    "proprio": {"min": [0.0] * 7, "max": [1.0] * 7, "q01": [0.0] * 7, "q99": [1.0] * 7}
                }
            }

        @classmethod
        def from_pretrained(cls, model_path, **kwargs):
            del model_path, kwargs
            fake_modeling_utils.dispatch_model(object(), device_map={"": 0})
            return cls()

        def eval(self):
            return self

        def to(self, device, dtype=None):
            del device, dtype
            return self

    def fake_dispatch_model(model, *args, **kwargs):
        del model, args
        calls["force_hooks"] = kwargs.get("force_hooks")
        return object()

    fake_processing.PrismaticImageProcessor = FakeImageProcessor
    fake_processing.PrismaticProcessor = FakeProcessor
    fake_modeling.OpenVLAForActionPrediction = FakeModel
    fake_transformers.AutoTokenizer = FakeTokenizer
    fake_transformers.BitsAndBytesConfig = FakeBitsAndBytesConfig
    fake_modeling_utils.dispatch_model = fake_dispatch_model

    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.processing_prismatic", fake_processing)
    monkeypatch.setitem(sys.modules, "prismatic.extern.hf.modeling_prismatic", fake_modeling)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)

    sys.modules.pop("roboneuron_core.runtime.openvla_oft_worker", None)
    openvla_oft_worker = importlib.import_module("roboneuron_core.runtime.openvla_oft_worker")

    monkeypatch.setattr(openvla_oft_worker, "_prepare_prismatic_shallow_packages", lambda: None)
    monkeypatch.setattr(
        openvla_oft_worker.OpenVLAOFTWorker,
        "_import_runtime_modules",
        lambda self: {
            "OpenVLAForActionPrediction": FakeModel,
            "PrismaticImageProcessor": FakeImageProcessor,
            "PrismaticProcessor": FakeProcessor,
            "L1RegressionActionHead": object,
            "DiffusionActionHead": object,
            "NoisyActionProjector": object,
            "ProprioProjector": object,
            "FiLMedPrismaticVisionBackbone": object,
        },
    )

    worker = openvla_oft_worker.OpenVLAOFTWorker(
        model_path=str(checkpoint_dir),
        attn_implementation="flash_attention_2",
        dtype_name="float16",
        device_name="cuda:0",
        runtime_quantization="4bit",
        low_cpu_mem_usage=True,
        use_l1_regression=False,
        use_diffusion=False,
        use_film=False,
        use_proprio=False,
        num_images_in_input=1,
        num_diffusion_steps_inference=50,
        lora_rank=32,
        center_crop=True,
        unnorm_key="vr_banana",
        robot_platform="bridge",
        default_proprio=None,
        base_model_path=None,
    )
    worker.load()

    assert calls["force_hooks"] is True
