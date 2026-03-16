from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from roboneuron_core.runtime.openvla_oft_client import OpenVLAOFTSubprocessClient


def _runtime_cuda_is_usable(runtime_python: Path) -> bool:
    probe = subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(); "
                "torch.cuda.device_count(); "
                "print('ok')"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0 and probe.stdout.strip() == "ok"


@pytest.mark.integration
def test_openvla_oft_real_checkpoint_smoke() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runtime_python = project_root / ".venvs" / "openvla-oft" / "bin" / "python"
    checkpoint_dir = project_root / "checkpoints" / "openvla-oft" / "openvla-oft-pick-banana"

    if not runtime_python.is_file():
        pytest.skip("OpenVLA-OFT runtime is not set up. Run scripts/setup_openvla_oft_runtime.sh first.")
    if not checkpoint_dir.is_dir():
        pytest.skip("OpenVLA-OFT checkpoint directory is missing.")

    runtime_device = os.getenv("OPENVLA_OFT_TEST_DEVICE")
    if runtime_device is None:
        runtime_device = "cuda:0" if shutil.which("nvidia-smi") else ""
    if not runtime_device:
        pytest.skip("No GPU detected. Set OPENVLA_OFT_TEST_DEVICE=cpu to force the smoke test on CPU.")
    if runtime_device.startswith("cuda") and not _runtime_cuda_is_usable(runtime_python):
        pytest.skip("CUDA runtime is not usable in this environment. Set OPENVLA_OFT_TEST_DEVICE=cpu if needed.")

    client = OpenVLAOFTSubprocessClient(
        model_path=checkpoint_dir,
        runtime_python=runtime_python,
        runtime_extra_python_paths=[
            str(project_root / "src"),
            str(project_root / "third_party" / "vla_src" / "openvla-oft"),
        ],
        startup_timeout_sec=float(os.getenv("OPENVLA_OFT_TEST_STARTUP_TIMEOUT", "3600")),
        request_timeout_sec=float(os.getenv("OPENVLA_OFT_TEST_REQUEST_TIMEOUT", "1200")),
        attn_implementation="flash_attention_2",
        device=runtime_device,
        unnorm_key="vr_banana",
        robot_platform="bridge",
        use_film=True,
        use_proprio=True,
        num_images_in_input=1,
        default_proprio=[0.0] * 7,
    )

    try:
        client.load()
        action = client.predict_action(
            observation={
                "full_image": Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8)),
                "state": np.zeros((7,), dtype=np.float32),
            },
            instruction="pick up banana",
            unnorm_key="vr_banana",
        )
    finally:
        client.close()

    assert action.ndim == 2
    assert action.shape[0] > 0
    assert action.shape[1] == 7
    assert np.isfinite(action).all()
