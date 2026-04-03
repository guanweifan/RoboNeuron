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
from PIL import Image

from .openvla_protocol import encode_image_to_base64

logger = logging.getLogger(__name__)


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def _resolve_runtime_path(path: str | Path, project_root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return project_root / candidate


class OpenVLASubprocessClient:
    """Manage a persistent OpenVLA worker living in its own Python runtime."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        runtime_python: str | Path | None = None,
        runtime_module: str = "roboneuron_core.runtime.openvla_worker",
        runtime_extra_python_paths: list[str] | tuple[str, ...] | None = None,
        startup_timeout_sec: float = 900.0,
        request_timeout_sec: float = 300.0,
        attn_implementation: str | None = None,
        dtype: str = "bfloat16",
        device: str = "auto",
        runtime_quantization: str = "none",
        low_cpu_mem_usage: bool = True,
    ) -> None:
        project_root = _project_root()
        default_runtime = project_root / ".venvs" / "openvla" / "bin" / "python"
        default_paths = [
            str(project_root / "src"),
            str(project_root / "third_party" / "vla_src" / "openvla"),
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
        self.runtime_quantization = runtime_quantization
        self.low_cpu_mem_usage = low_cpu_mem_usage
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
            setup_script = self.project_root / "scripts" / "setup_openvla_runtime.sh"
            raise RuntimeError(
                f"OpenVLA runtime python not found: {runtime_python}. "
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
            "--runtime-quantization",
            self.runtime_quantization,
        ]
        if self.attn_implementation is not None:
            cmd.extend(["--attn-implementation", self.attn_implementation])
        if self.low_cpu_mem_usage:
            cmd.append("--low-cpu-mem-usage")

        env = os.environ.copy()
        python_paths = list(self.runtime_extra_python_paths)
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            python_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

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
            raise RuntimeError(f"OpenVLA worker failed during startup: {ready}")

    def predict_action(
        self,
        *,
        image: Image.Image,
        instruction: str,
        unnorm_key: str | None,
        extra_predict_kwargs: dict[str, Any] | None = None,
    ) -> np.ndarray:
        payload = {
            "image_base64": encode_image_to_base64(image),
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
                logger.debug("OpenVLA worker did not acknowledge shutdown; terminating.", exc_info=True)
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
                logger.error("Non-JSON stdout from OpenVLA worker: %s", raw)

        self._messages.put({"event": "eof", "returncode": self._process.poll()})

    def _stderr_pump(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None

        for line in self._process.stderr:
            raw = line.rstrip()
            if raw:
                logger.info("[openvla-runtime] %s", raw)

    def _request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        if not self.is_running or self._process is None:
            raise RuntimeError("OpenVLA runtime is not running. Call load() first.")

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
                    f"OpenVLA worker exited unexpectedly with return code {message.get('returncode')}."
                )
            if message.get("id") != request_id:
                raise RuntimeError(f"Out-of-order response from OpenVLA worker: {message}")
            if not message.get("ok", False):
                error = message.get("error", {})
                details = (
                    "OpenVLA worker error: "
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
            raise TimeoutError("Timed out waiting for response from OpenVLA worker.") from exc
