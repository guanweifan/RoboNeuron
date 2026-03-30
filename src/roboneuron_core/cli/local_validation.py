"""Stable local validation entrypoint for day-to-day RoboNeuron development."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

LOCAL_VALIDATION_TESTS = (
    "tests/test_action_semantics.py",
    "tests/test_control_runtime.py",
    "tests/test_control_server_host_status.py",
    "tests/test_core_edge_split_smoke.py",
    "tests/test_distributed_core_edge_lane_smoke.py",
    "tests/test_dummy_adapters.py",
    "tests/test_kernel_primitives.py",
    "tests/test_local_validation_cli.py",
    "tests/test_mcp_cli_entrypoints.py",
    "tests/test_openvla_oft_subprocess_client.py",
    "tests/test_openvla_oft_wrapper_runtime.py",
    "tests/test_openvla_subprocess_client.py",
    "tests/test_openvla_wrapper_runtime.py",
    "tests/test_perception_server.py",
    "tests/test_raw_action_chunk_utils.py",
    "tests/test_task_space_state_utils.py",
    "tests/test_vla_model_resolution.py",
    "tests/test_vla_server_dummy.py",
    "tests/test_vla_server_local_test.py",
    "tests/test_vla_server_observation_building.py",
)


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def build_validation_command(*, python_executable: str | None = None) -> list[str]:
    return [
        python_executable or sys.executable,
        "-m",
        "pytest",
        "-q",
        *LOCAL_VALIDATION_TESTS,
    ]


def run_local_validation() -> int:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    result = subprocess.run(
        build_validation_command(),
        cwd=_project_root(),
        env=env,
        check=False,
    )
    return result.returncode


def main() -> None:
    raise SystemExit(run_local_validation())
