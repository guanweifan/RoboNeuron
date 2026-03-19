#!/usr/bin/env python3

"""MCP server for managing VLA inference nodes."""

import json
import multiprocessing
from contextlib import suppress
from pathlib import Path
from typing import Any

import json_numpy
import numpy as np
import torch
from mcp.server.fastmcp import FastMCP
from PIL import Image

from roboneuron_core.adapters.vla import get_registry
from roboneuron_core.adapters.vla.dummy_vla import DUMMY_MODEL_PATH
from roboneuron_core.utils.control_runtime import DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL
from roboneuron_core.utils.raw_action_chunk import RAW_ACTION_CHUNK_TOPIC

_VLA_PROCESS: multiprocessing.Process | None = None
EEF_DELTA_CMD_TOPIC = "/eef_delta_cmd"
mcp = FastMCP("roboneuron-vla")


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def _load_vla_models_config() -> dict[str, dict[str, Any]]:
    """Load normalized model configuration from `configs/vla_models.json`."""
    project_root = _project_root()
    cfg_path = project_root / "configs" / "vla_models.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"VLA models config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vla_models.json must be a JSON object mapping names->configs, got {type(data)}")

    models: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if isinstance(v, str):
            models[k] = {"path": v, "kwargs": {}}
        elif isinstance(v, dict) and isinstance(v.get("path"), str):
            wrapper_kwargs = dict(v.get("kwargs", {})) if isinstance(v.get("kwargs", {}), dict) else {}
            for key, value in v.items():
                if key not in {"path", "kwargs"}:
                    wrapper_kwargs[key] = value
            models[k] = {"path": v["path"], "kwargs": wrapper_kwargs}
        else:
            continue
    return models


def _resolve_model_spec(model_name: str, model_path: str | None) -> tuple[str, dict[str, Any]]:
    """Resolve model path and wrapper kwargs while allowing dummy adapters to run without checkpoints."""
    if model_name == "dummy" and not model_path:
        return DUMMY_MODEL_PATH, {}

    models_cfg = _load_vla_models_config()
    cfg_entry = models_cfg.get(model_name)

    if model_path:
        return model_path, dict(cfg_entry.get("kwargs", {})) if isinstance(cfg_entry, dict) else {}

    if not isinstance(cfg_entry, dict):
        raise ValueError(
            f"model_path not provided and model '{model_name}' not found in configs/vla_models.json."
            " Create configs/vla_models.json with a mapping or pass --model-path."
        )
    return str(cfg_entry["path"]), dict(cfg_entry.get("kwargs", {}))


def _resolve_output_contract(
    model_name: str,
    output_mode: str,
    action_protocol: str | None,
    action_frame: str,
) -> tuple[str, str, str]:
    """Resolve the startup-time ROS output contract for a VLA node."""

    valid_output_modes = {"auto", "eef_delta", "raw_action_chunk"}
    if output_mode not in valid_output_modes:
        raise ValueError(
            f"Unsupported output_mode '{output_mode}'. Expected one of: {sorted(valid_output_modes)}."
        )

    if output_mode == "auto":
        resolved_mode = "raw_action_chunk" if model_name == "openvla-oft" else "eef_delta"
    else:
        resolved_mode = output_mode

    if resolved_mode == "eef_delta":
        return resolved_mode, "eef_delta", action_frame

    resolved_protocol = action_protocol or DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL
    return resolved_mode, resolved_protocol, action_frame


def _resolve_output_topic(output_topic: str, resolved_output_mode: str) -> str:
    """Use protocol-appropriate default topics while keeping explicit overrides intact."""

    if resolved_output_mode == "raw_action_chunk" and output_topic == EEF_DELTA_CMD_TOPIC:
        return RAW_ACTION_CHUNK_TOPIC
    return output_topic


def _build_model_observation(
    *,
    model_name: str,
    primary_image: Image.Image,
    instruction: str,
    wrist_image: Image.Image | None = None,
    task_space_state: np.ndarray | None = None,
) -> tuple[Image.Image | dict[str, Any], dict[str, Any]]:
    """Build the per-model observation payload from cached ROS inputs."""

    if model_name != "openvla-oft":
        return primary_image, {}

    observation: dict[str, Any] = {
        "full_image": primary_image,
        "instruction": instruction,
    }
    if wrist_image is not None:
        observation["wrist_image"] = wrist_image
    if task_space_state is not None:
        observation["state"] = np.asarray(task_space_state, dtype=np.float32)
    return observation, {}


def _load_ros_runtime() -> tuple[Any, type[Any]]:
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from roboneuron_interfaces.msg import EEFDeltaCommand, RawActionChunk, TaskSpaceState
        from sensor_msgs.msg import Image as RosImage

        from roboneuron_core.utils.eef_delta import array_to_eef_delta_command
        from roboneuron_core.utils.raw_action_chunk import array_to_raw_action_chunk_message
        from roboneuron_core.utils.task_space_state import task_space_state_message_to_array
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ROS Python packages are not available in this interpreter. "
            "Source `/opt/ros/humble/setup.bash` and your ROS workspace, and do not overwrite "
            "`PYTHONPATH`. Use `PYTHONPATH=src:$PYTHONPATH` or run "
            "`python -m roboneuron_core.servers.vla_server` from the repo root."
        ) from exc

    class VLAServerNode(Node):
        """ROS 2 node that runs the VLA inference loop."""

        def __init__(
            self,
            model_name: str,
            model_path: str,
            model_kwargs: dict[str, Any] | None,
            input_topic: str,
            output_topic: str,
            instruction: str = "do task",
            output_mode: str = "eef_delta",
            action_protocol: str = "eef_delta",
            action_frame: str = "tool",
            step_duration_sec: float = 0.1,
            wrist_input_topic: str | None = None,
            state_topic: str | None = None,
        ):
            super().__init__("vla_server_node")
            self._model_name = model_name
            self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            self._instruction = instruction
            self._output_mode = output_mode
            self._action_protocol = action_protocol
            self._action_frame = action_frame
            self._step_duration_sec = step_duration_sec
            self._latest_wrist_image: Image.Image | None = None
            self._latest_task_space_state: np.ndarray | None = None

            json_numpy.patch()
            registry = get_registry()
            wrapper_class = registry[model_name]
            wrapper_kwargs = {"device": self._device}
            if model_kwargs:
                wrapper_kwargs.update(model_kwargs)
            self._model = wrapper_class(model_path, **wrapper_kwargs)
            self._model.load()

            try:
                dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
                action = self._model.predict_action(
                    image=dummy_img,
                    instruction=self._instruction,
                )
                self.get_logger().info(f"[SelfTest] dummy action shape={getattr(action, 'shape', None)}")
            except Exception as e:
                self.get_logger().error(f"[SelfTest] dummy predict_action failed: {e}")

            self._cv_bridge = CvBridge()
            self._sub = self.create_subscription(RosImage, input_topic, self._image_cb, 10)
            if wrist_input_topic:
                self._wrist_sub = self.create_subscription(RosImage, wrist_input_topic, self._wrist_image_cb, 10)
            else:
                self._wrist_sub = None
            if state_topic:
                self._state_sub = self.create_subscription(TaskSpaceState, state_topic, self._state_cb, 10)
            else:
                self._state_sub = None
            if self._output_mode == "eef_delta":
                self._pub = self.create_publisher(EEFDeltaCommand, output_topic, 10)
            else:
                self._pub = self.create_publisher(RawActionChunk, output_topic, 10)

            self.get_logger().info(
                f"VLA Model {model_name} listening on {input_topic} "
                f"and publishing {self._output_mode} on {output_topic}"
            )
            if wrist_input_topic:
                self.get_logger().info(f"Using wrist image topic: {wrist_input_topic}")
            if state_topic:
                self.get_logger().info(f"Using task-space state topic: {state_topic}")

        def _wrist_image_cb(self, msg: RosImage) -> None:
            try:
                cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                self._latest_wrist_image = Image.fromarray(cv_img)
            except Exception as exc:
                self.get_logger().error(f"Failed to decode wrist image: {exc}")

        def _state_cb(self, msg: TaskSpaceState) -> None:
            try:
                self._latest_task_space_state = task_space_state_message_to_array(msg)
            except Exception as exc:
                self.get_logger().error(f"Failed to decode task-space state: {exc}")

        def _image_cb(self, msg: RosImage) -> None:
            """Run inference on each incoming image and publish the predicted action."""
            try:
                cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                pil_img = Image.fromarray(cv_img)

                observation_input, predict_kwargs = _build_model_observation(
                    model_name=self._model_name,
                    primary_image=pil_img,
                    instruction=self._instruction,
                    wrist_image=self._latest_wrist_image,
                    task_space_state=self._latest_task_space_state,
                )
                action = self._model.predict_action(
                    image=observation_input,
                    instruction=self._instruction,
                    **predict_kwargs,
                )

                if action is not None:
                    action_np = np.asarray(action, dtype=np.float64)
                    if self._output_mode == "eef_delta":
                        out_msg = array_to_eef_delta_command(action_np.reshape(-1).tolist())
                    else:
                        out_msg = array_to_raw_action_chunk_message(
                            action_np,
                            protocol=self._action_protocol,
                            step_duration_sec=self._step_duration_sec,
                            frame=self._action_frame,
                        )
                    self._pub.publish(out_msg)
            except Exception as exc:
                self.get_logger().error(f"VLA inference callback failed: {exc}")

    return rclpy, VLAServerNode


def _run_local_test(
    model_name: str,
    model_path: str | None,
    instruction: str,
) -> int:
    import numpy as np

    registry = get_registry()
    if model_name not in registry:
        print(f"Error: model '{model_name}' not found in registry.")
        return 1

    try:
        resolved_model_path, model_kwargs = _resolve_model_spec(model_name, model_path)
    except Exception as exc:
        print(f"Error: failed to resolve model spec: {exc}")
        return 1

    wrapper_class = registry[model_name]
    wrapper_kwargs = {"device": torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")}
    wrapper_kwargs.update(model_kwargs)
    model = wrapper_class(resolved_model_path, **wrapper_kwargs)

    print(f"LOCAL TEST MODE: loading {model_name} from {resolved_model_path}")
    try:
        model.load()
        dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        action = model.predict_action(
            image=dummy_img,
            instruction=instruction,
        )
        print(f"Local test succeeded. action shape={getattr(action, 'shape', None)} action={action}")
        return 0
    except Exception as exc:
        print(f"Local test failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        with suppress(Exception):
            close = getattr(model, "close", None)
            if callable(close):
                close()


def _ros_worker(
    model_name: str,
    model_path: str,
    model_kwargs: dict[str, Any] | None,
    input_topic: str,
    output_topic: str,
    instruction: str,
    output_mode: str,
    action_protocol: str,
    action_frame: str,
    step_duration_sec: float,
    wrist_input_topic: str | None,
    state_topic: str | None,
):
    """Multiprocessing worker function to initialize and run the VLA inference node."""
    rclpy, vla_server_node_class = _load_ros_runtime()
    rclpy.init()
    node = vla_server_node_class(
        model_name,
        model_path,
        model_kwargs,
        input_topic,
        output_topic,
        instruction=instruction,
        output_mode=output_mode,
        action_protocol=action_protocol,
        action_frame=action_frame,
        step_duration_sec=step_duration_sec,
        wrist_input_topic=wrist_input_topic,
        state_topic=state_topic,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        if type(exc).__name__ != "ExternalShutdownException":
            raise
    finally:
        with suppress(Exception):
            node.destroy_node()
        with suppress(Exception):
            rclpy.shutdown()


@mcp.tool()
def start_vla_inference(
    model_name: str,
    model_path: str | None,
    instruction: str,
    input_topic: str = "/isaac_rgb",
    output_topic: str = EEF_DELTA_CMD_TOPIC,
    output_mode: str = "auto",
    action_protocol: str | None = None,
    action_frame: str = "tool",
    step_duration_sec: float = 0.1,
    wrist_input_topic: str | None = None,
    state_topic: str | None = None,
) -> str:
    """
    [VLA REASONING] Start the Vision-Language-Action node for instruction-conditioned control.

    The node subscribes to visual input and publishes end-effector delta commands to the control topic.

    REQUIRED TASK FLOW (To pick up an object):
    1. Call `start_camera` (to provide images to input_topic).
    2. Call `start_controller` (to make the robot listen on output_topic).
    3. Call this tool with the detailed 'instruction'.

    Args:
        model_name (str): The VLA model identifier (`dummy`, `openvla`, or `openvla-oft`).
        model_path (str|None): Path to the model checkpoint. If None, attempt to read from configs/vla_models.json.
        instruction (str): The natural language instruction defining the task (e.g., "pick up the blue bowl").
        input_topic (str): [Input Topic] Topic subscribed for images (must match camera output).
        output_topic (str): [Output Topic] Topic used to publish VLA-predicted actions (must match controller input).
        output_mode (str): `eef_delta`, `raw_action_chunk`, or `auto`.
        action_protocol (str|None): Semantic protocol name for raw action chunks.
        action_frame (str): Reference frame for raw Cartesian actions.
        step_duration_sec (float): Per-step duration used when publishing raw action chunks.
        wrist_input_topic (str|None): Optional wrist camera topic for OpenVLA-OFT style observations.
        state_topic (str|None): Optional 7D task-space state topic for OpenVLA-OFT style observations.
    """
    global _VLA_PROCESS
    if _VLA_PROCESS is not None and _VLA_PROCESS.is_alive():
        return "Error: VLA is already running. Stop it first."

    # Validate model key early (registry)
    try:
        registry = get_registry()
        if model_name not in registry:
            return f"Error: model '{model_name}' not found in registry."
    except Exception as e:
        return f"Error: failed to access model registry: {e}"

    try:
        model_path, model_kwargs = _resolve_model_spec(model_name, model_path)
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as e:
        return f"Error: failed to load vla models config: {e}"

    try:
        resolved_output_mode, resolved_action_protocol, resolved_action_frame = _resolve_output_contract(
            model_name,
            output_mode,
            action_protocol,
            action_frame,
        )
    except ValueError as exc:
        return f"Error: {exc}"

    resolved_output_topic = _resolve_output_topic(output_topic, resolved_output_mode)

    if step_duration_sec <= 0:
        return "Error: step_duration_sec must be positive."

    try:
        _load_ros_runtime()
    except RuntimeError as exc:
        return f"Error: {exc}"

    # Use spawn to avoid fork-related rclpy issues
    ctx = multiprocessing.get_context("spawn")
    _VLA_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(
            model_name,
            model_path,
            model_kwargs,
            input_topic,
            resolved_output_topic,
            instruction,
            resolved_output_mode,
            resolved_action_protocol,
            resolved_action_frame,
            step_duration_sec,
            wrist_input_topic,
            state_topic,
        ),
        daemon=False,
    )
    _VLA_PROCESS.start()
    return f"Success: VLA {model_name} started (pid={_VLA_PROCESS.pid})."


@mcp.tool()
def stop_vla_inference() -> str:
    """Stops the running VLA inference loop and cleans up the background process."""
    global _VLA_PROCESS
    if _VLA_PROCESS is None or not _VLA_PROCESS.is_alive():
        return "Info: No VLA is running."

    _VLA_PROCESS.terminate()
    _VLA_PROCESS.join(timeout=5.0)
    if _VLA_PROCESS.is_alive():
        with suppress(Exception):
            _VLA_PROCESS.kill()
        _VLA_PROCESS.join(timeout=1.0)

    _VLA_PROCESS = None
    return "Success: VLA stopped."


if __name__ == "__main__":
    # Local test harness
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="vla_server.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--run-ros", action="store_true", help="Run the ROS worker in the foreground instead of spawning from MCP")
    parser.add_argument("--model-name", type=str, default="dummy", help="Model registry key to test")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--instruction", type=str, default="do task", help="Natural language instruction for the VLA task")
    parser.add_argument("--input-topic", type=str, default="/isaac_rgb", help="Input image topic")
    parser.add_argument("--output-topic", type=str, default=EEF_DELTA_CMD_TOPIC, help="Output action topic")
    parser.add_argument("--output-mode", type=str, default="auto", choices=["auto", "eef_delta", "raw_action_chunk"], help="Output action contract")
    parser.add_argument("--action-protocol", type=str, default=None, help="Protocol name for raw action chunks")
    parser.add_argument("--action-frame", type=str, default="tool", help="Reference frame for raw action chunks")
    parser.add_argument("--step-duration-sec", type=float, default=0.1, help="Per-step duration for raw action chunks")
    parser.add_argument("--wrist-input-topic", type=str, default=None, help="Optional wrist camera topic for OpenVLA-OFT observations")
    parser.add_argument("--state-topic", type=str, default=None, help="Optional 7D task-space state topic for OpenVLA-OFT observations")
    args = parser.parse_args()

    if args.local_test:
        sys.exit(
            _run_local_test(
                args.model_name,
                args.model_path,
                args.instruction,
            )
        )
    if args.run_ros:
        try:
            model_path, model_kwargs = _resolve_model_spec(args.model_name, args.model_path)
            resolved_output_mode, resolved_action_protocol, resolved_action_frame = _resolve_output_contract(
                args.model_name,
                args.output_mode,
                args.action_protocol,
                args.action_frame,
            )
            resolved_output_topic = _resolve_output_topic(args.output_topic, resolved_output_mode)
        except Exception as exc:
            print(f"Error: {exc}")
            sys.exit(1)

        _ros_worker(
            args.model_name,
            model_path,
            model_kwargs,
            args.input_topic,
            resolved_output_topic,
            args.instruction,
            resolved_output_mode,
            resolved_action_protocol,
            resolved_action_frame,
            args.step_duration_sec,
            args.wrist_input_topic,
            args.state_topic,
        )
    else:
        # Run as an MCP service
        mcp.run()
