from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .openvla_client import _project_root, _resolve_runtime_path
from .openvla_oft_protocol import encode_observation_for_transport

logger = logging.getLogger(__name__)


class OpenVLAOFTSubprocessClient:
    """Manage a persistent OpenVLA-OFT worker living in its own Python runtime."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        runtime_python: str | Path | None = None,
        runtime_module: str = "roboneuron_core.runtime.openvla_oft_worker",
        runtime_extra_python_paths: list[str] | tuple[str, ...] | None = None,
        startup_timeout_sec: float = 1800.0,
        request_timeout_sec: float = 300.0,
        attn_implementation: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
        low_cpu_mem_usage: bool = True,
        use_l1_regression: bool | None = None,
        use_diffusion: bool | None = None,
        use_film: bool | None = None,
        use_proprio: bool | None = None,
        num_images_in_input: int | None = None,
        num_diffusion_steps_inference: int = 50,
        lora_rank: int = 32,
        center_crop: bool = True,
        unnorm_key: str | None = None,
        robot_platform: str | None = None,
        default_proprio: list[float] | tuple[float, ...] | np.ndarray | None = None,
        base_model_path: str | Path | None = None,
    ) -> None:
        project_root = _project_root()
        default_runtime = project_root / ".venvs" / "openvla-oft" / "bin" / "python"
        default_paths = [
            str(project_root / "src"),
            str(project_root / "third_party" / "vla_src" / "openvla-oft"),
        ]
        resolved_runtime_python = _resolve_runtime_path(runtime_python or default_runtime, project_root)
        resolved_runtime_extra_python_paths = [
            str(_resolve_runtime_path(path, project_root))
            for path in (runtime_extra_python_paths or default_paths)
        ]

        self.model_path = str(_resolve_runtime_path(model_path, project_root))
        self.runtime_python = str(resolved_runtime_python)
        self.runtime_module = runtime_module
        self.runtime_extra_python_paths = resolved_runtime_extra_python_paths
        self.startup_timeout_sec = startup_timeout_sec
        self.request_timeout_sec = request_timeout_sec
        self.attn_implementation = attn_implementation
        self.dtype = dtype
        self.device = device
        self.low_cpu_mem_usage = low_cpu_mem_usage
        self.use_l1_regression = use_l1_regression
        self.use_diffusion = use_diffusion
        self.use_film = use_film
        self.use_proprio = use_proprio
        self.num_images_in_input = num_images_in_input
        self.num_diffusion_steps_inference = num_diffusion_steps_inference
        self.lora_rank = lora_rank
        self.center_crop = center_crop
        self.unnorm_key = unnorm_key
        self.robot_platform = robot_platform
        self.default_proprio = (
            np.asarray(default_proprio, dtype=np.float32).tolist() if default_proprio is not None else None
        )
        self.base_model_path = (
            str(_resolve_runtime_path(base_model_path, project_root)) if base_model_path is not None else None
        )
        self.project_root = project_root

        self._process: subprocess.Popen[str] | None = None
        self._messages: queue.Queue[dict[str, Any]] = queue.Queue()
        self._request_id = 0
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def load(self) -> None:
        if self.is_running:
            return

        runtime_python = Path(self.runtime_python)
        if not runtime_python.is_file():
            setup_script = self.project_root / "scripts" / "setup_openvla_oft_runtime.sh"
            raise RuntimeError(
                f"OpenVLA-OFT runtime python not found: {runtime_python}. "
                f"Create the dedicated runtime first, e.g. `{setup_script}`."
            )

        cmd = [
            self.runtime_python,
            "-m",
            self.runtime_module,
            "--model-path",
            self.model_path,
            "--dtype",
            self.dtype,
            "--device",
            self.device,
            "--num-diffusion-steps-inference",
            str(self.num_diffusion_steps_inference),
            "--lora-rank",
            str(self.lora_rank),
        ]
        if self.attn_implementation is not None:
            cmd.extend(["--attn-implementation", self.attn_implementation])
        if self.low_cpu_mem_usage:
            cmd.append("--low-cpu-mem-usage")
        if self.use_l1_regression is not None:
            cmd.extend(["--use-l1-regression", str(self.use_l1_regression).lower()])
        if self.use_diffusion is not None:
            cmd.extend(["--use-diffusion", str(self.use_diffusion).lower()])
        if self.use_film is not None:
            cmd.extend(["--use-film", str(self.use_film).lower()])
        if self.use_proprio is not None:
            cmd.extend(["--use-proprio", str(self.use_proprio).lower()])
        if self.num_images_in_input is not None:
            cmd.extend(["--num-images-in-input", str(self.num_images_in_input)])
        if self.center_crop:
            cmd.append("--center-crop")
        else:
            cmd.append("--no-center-crop")
        if self.unnorm_key is not None:
            cmd.extend(["--unnorm-key", self.unnorm_key])
        if self.robot_platform is not None:
            cmd.extend(["--robot-platform", self.robot_platform])
        if self.default_proprio is not None:
            cmd.extend(["--default-proprio-json", json.dumps(self.default_proprio, separators=(",", ":"))])
        if self.base_model_path is not None:
            cmd.extend(["--base-model-path", self.base_model_path])

        env = os.environ.copy()
        python_paths = list(self.runtime_extra_python_paths)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        hf_home = self.project_root / ".cache" / "huggingface"
        hf_hub_cache = hf_home / "hub"
        hf_modules_cache = hf_home / "modules"
        hf_hub_cache.mkdir(parents=True, exist_ok=True)
        hf_modules_cache.mkdir(parents=True, exist_ok=True)
        env.setdefault("HF_HOME", str(hf_home))
        env.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_hub_cache))
        env.setdefault("TRANSFORMERS_CACHE", str(hf_hub_cache))
        env.setdefault("HF_MODULES_CACHE", str(hf_modules_cache))

        self._process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._start_reader_threads()

        ready = self._wait_for_message(timeout=self.startup_timeout_sec)
        if ready.get("event") != "ready":
            raise RuntimeError(f"OpenVLA-OFT worker failed during startup: {ready}")

    def predict_action(
        self,
        *,
        observation: dict[str, Any],
        instruction: str,
        unnorm_key: str | None,
        extra_predict_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        payload = {
            "observation": encode_observation_for_transport(observation),
            "instruction": instruction,
            "unnorm_key": unnorm_key,
            "kwargs": dict(extra_predict_kwargs or {}),
        }
        response = self._request("predict_action", payload)
        action = response["result"]["action"]
        return np.asarray(action, dtype=np.float32)

    def close(self) -> None:
        process = self._process
        if process is None:
            return

        if process.poll() is None:
            try:
                self._request("shutdown", {}, timeout=10.0)
            except Exception:
                logger.debug("OpenVLA-OFT worker did not acknowledge shutdown; terminating.", exc_info=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)

        self._process = None

    def _start_reader_threads(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None

        threading.Thread(target=self._stdout_pump, daemon=True).start()
        threading.Thread(target=self._stderr_pump, daemon=True).start()

    def _stdout_pump(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None

        for line in self._process.stdout:
            raw = line.strip()
            if not raw:
                continue
            try:
                self._messages.put(json.loads(raw))
            except json.JSONDecodeError:
                logger.error("Non-JSON stdout from OpenVLA-OFT worker: %s", raw)

        self._messages.put({"event": "eof", "returncode": self._process.poll()})

    def _stderr_pump(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None

        for line in self._process.stderr:
            raw = line.rstrip()
            if raw:
                logger.info("[openvla-oft-runtime] %s", raw)

    def _request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self.is_running or self._process is None:
            raise RuntimeError("OpenVLA-OFT runtime is not running. Call load() first.")

        with self._lock:
            self._request_id += 1
            request_id = self._request_id

            assert self._process.stdin is not None
            self._process.stdin.write(
                json.dumps({"id": request_id, "method": method, "params": params}, separators=(",", ":")) + "\n"
            )
            self._process.stdin.flush()

            message = self._wait_for_message(timeout=timeout or self.request_timeout_sec)
            if message.get("event") == "eof":
                raise RuntimeError(
                    f"OpenVLA-OFT worker exited unexpectedly with return code {message.get('returncode')}."
                )
            if message.get("id") != request_id:
                raise RuntimeError(f"Out-of-order response from OpenVLA-OFT worker: {message}")
            if not message.get("ok", False):
                error = message.get("error", {})
                details = (
                    "OpenVLA-OFT worker error: "
                    f"{error.get('type', 'RuntimeError')}: {error.get('message', 'unknown error')}"
                )
                traceback_text = error.get("traceback")
                if traceback_text:
                    details = f"{details}\n{traceback_text}"
                raise RuntimeError(details)
            return message

    def _wait_for_message(self, *, timeout: float) -> dict[str, Any]:
        try:
            return self._messages.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for response from OpenVLA-OFT worker.") from exc
