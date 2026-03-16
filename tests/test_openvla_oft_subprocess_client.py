from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from roboneuron_core.runtime.openvla_oft_client import OpenVLAOFTSubprocessClient


def test_openvla_oft_subprocess_client_round_trip() -> None:
    project_root = Path(__file__).resolve().parents[1]
    client = OpenVLAOFTSubprocessClient(
        model_path="checkpoints/openvla-oft/openvla-oft-pick-banana",
        runtime_python="/usr/bin/python3",
        runtime_module="fake_openvla_oft_worker",
        runtime_extra_python_paths=[
            str(project_root / "src"),
            str(project_root / "tests"),
        ],
        startup_timeout_sec=10.0,
        request_timeout_sec=10.0,
        dtype="float32",
        device="cpu",
        low_cpu_mem_usage=False,
        base_model_path="checkpoints/openvla/openvla-7b",
    )
    assert client.model_path == str(project_root / "checkpoints" / "openvla-oft" / "openvla-oft-pick-banana")
    assert client.base_model_path == str(project_root / "checkpoints" / "openvla" / "openvla-7b")
    assert client.runtime_extra_python_paths == [
        str(project_root / "src"),
        str(project_root / "tests"),
    ]

    try:
        client.load()
        action = client.predict_action(
            observation={
                "full_image": Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
                "state": np.zeros((7,), dtype=np.float32),
            },
            instruction="pick up banana",
            unnorm_key="vr_banana",
        )
    finally:
        client.close()

    np.testing.assert_allclose(action, np.array([[1.0, 7.0, 1.0]], dtype=np.float32))
