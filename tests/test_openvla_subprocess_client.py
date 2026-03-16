from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from roboneuron_core.runtime.openvla_client import OpenVLASubprocessClient


def test_openvla_subprocess_client_round_trip() -> None:
    project_root = Path(__file__).resolve().parents[1]
    client = OpenVLASubprocessClient(
        model_path="checkpoints/openvla/openvla-7b",
        runtime_python=".venv/bin/python",
        runtime_module="fake_openvla_worker",
        runtime_extra_python_paths=[
            str(project_root / "src"),
            str(project_root / "tests"),
        ],
        startup_timeout_sec=10.0,
        request_timeout_sec=10.0,
        dtype="float32",
        device="cpu",
        low_cpu_mem_usage=False,
    )
    assert client.model_path == str(project_root / "checkpoints" / "openvla" / "openvla-7b")
    assert client.runtime_extra_python_paths == [
        str(project_root / "src"),
        str(project_root / "tests"),
    ]

    try:
        client.load()
        action = client.predict_action(
            image=Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)),
            instruction="pick up object",
            unnorm_key="bridge_orig",
        )
    finally:
        client.close()

    np.testing.assert_allclose(action, np.array([[14.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
