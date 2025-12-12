#!/usr/bin/env python3

"""
infer_mcp.py

MCP Server for managing VLA (Vision-Language Action) Inference nodes.
Allows the LLM to load specific models and bind them to specific input topics.

Acceleration Design Parameters:
- accel_method: Name of the specific acceleration technique (e.g., "none", "fastv", "quant", "sparse").
- accel_level: Specific preset or tier name (e.g., "off", "balanced", "aggressive", or "int8").
- Configuration is resolved from 'vla_accel_presets.json' based on [accel_method][accel_level][model_name/default].
"""

import multiprocessing
from pathlib import Path
import json
from typing import Optional, Tuple, Dict, Any, List
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
from PIL import Image
import torch
import json_numpy
from mcp.server.fastmcp import FastMCP

# Ensure wrapper imports are available
from wrapper.vla_wrapper import get_registry

_VLA_PROCESS: Optional[multiprocessing.Process] = None
mcp = FastMCP("robomcp-vla")


# ================= Acceleration Configuration Loading & Resolution ================= #

def _load_accel_config_file() -> Dict[str, Any]:
    """
    Loads the acceleration configuration from 'vla_accel_presets.json'.

    Raises:
        FileNotFoundError: If the configuration file is missing.
        ValueError: If the file content is not a valid JSON dictionary.
    """
    # Navigate to project root, then to the config directory
    cfg_path = Path(__file__).parent.parent / "config" / "vla_accel_presets.json"

    if not cfg_path.is_file():
        raise FileNotFoundError(
            f"Acceleration config file not found: {cfg_path}. "
            f"Please create vla_accel_presets.json next to infer_mcp.py."
        )

    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            f"Acceleration config must be a JSON object (dict), "
            f"got {type(data)} from {cfg_path}."
        )

    return data


def resolve_accel_configs(
    model_name: str,
    accel_method: str,
    accel_level: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Resolves method-specific and pruning configurations based on model name, method, and level.

    Configuration lookup priority: [method][level][model_name] -> [method][level]["default"].

    Args:
        model_name (str): The name of the VLA model (e.g., "OpenVLA").
        accel_method (str): The acceleration method (e.g., "fastv").
        accel_level (str): The acceleration preset level (e.g., "balanced").

    Returns:
        Tuple[Optional[dict], Optional[dict]]: (method_config, prune_config).
    """
    config = _load_accel_config_file()

    method_cfg = config.get(accel_method, {})
    level_cfg = method_cfg.get(accel_level, {})

    target_cfg = level_cfg.get(model_name)
    if not isinstance(target_cfg, dict):
        target_cfg = level_cfg.get("default", {})

    if not isinstance(target_cfg, dict):
        return None, None

    method_config = target_cfg.get("method_config")
    prune_config = target_cfg.get("prune_config")
    return method_config, prune_config


# ---------- vla model list loader ----------
def _load_vla_models_config() -> Dict[str, str]:
    """
    Loads model-name -> model-path mapping from 'config/vla_models.json'.
    Returns a dict mapping model_name -> model_path.

    Raises FileNotFoundError or ValueError on bad format.
    """
    cfg_path = Path(__file__).parent.parent / "config" / "vla_models.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"VLA models config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"vla_models.json must be a JSON object mapping names->paths, got {type(data)}")
    # allow simple nested objects but prefer string path values
    models: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, str):
            models[k] = v
        elif isinstance(v, dict) and "path" in v and isinstance(v["path"], str):
            models[k] = v["path"]
        else:
            # ignore malformed entries
            continue
    return models
# ---------- end ----------


# ================= ROS Node ================= #

class VLAServerNode(Node):
    """
    ROS2 Node implementing the VLA inference loop.

    It loads the VLA model, subscribes to an image topic, performs inference,
    and publishes the resulting action command.
    """
    def __init__(
        self,
        model_name: str,
        model_path: str,
        input_topic: str,
        output_topic: str,
        accel_method: str = "none",
        accel_level: str = "off",
        instruction: str = "do task",
    ):
        super().__init__("vla_server_node")
        self._device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self._accel_method = accel_method
        self._accel_level = accel_level
        self._instruction = instruction

        # 1) Resolve Acceleration Configuration
        try:
            self._method_config, self._prune_config = resolve_accel_configs(
                model_name=model_name,
                accel_method=accel_method,
                accel_level=accel_level,
            )
        except Exception as e:
            self.get_logger().error(f"Failed to resolve accel configs: {e}")
            self._method_config = None
            self._prune_config = None

        # 2) Load Model Wrapper
        json_numpy.patch()
        registry = get_registry()
        WrapperClass = registry[model_name]
        wrapper_kwargs = {
            "device": self._device,
            "accel_method": accel_method,
            "accel_level": accel_level,
        }
        self._model = WrapperClass(model_path, **wrapper_kwargs)
        self._model.load()

        # Self-Test for model and acceleration setup
        try:
            import numpy as np
            dummy_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))

            self.get_logger().info(
                f"[SelfTest] accel_method={self._accel_method}, "
                f"accel_config={self._method_config}, "
                f"prune_config={self._prune_config}"
            )

            action = self._model.predict_action(
                image=dummy_img,
                instruction=self._instruction,
                accel_method=self._accel_method,
                accel_config=self._method_config,
                prune_config=self._prune_config,
            )

            self.get_logger().info(f"[SelfTest] dummy action shape={getattr(action, 'shape', None)}")
        except Exception as e:
            self.get_logger().error(f"[SelfTest] dummy predict_action failed: {e}")

        # 3) ROS Subscription / Publication Setup
        self._cv_bridge = CvBridge()
        self._sub = self.create_subscription(RosImage, input_topic, self._image_cb, 10)
        self._pub = self.create_publisher(Float64MultiArray, output_topic, 10)

        self.get_logger().info(
            f"VLA Model {model_name} listening on {input_topic} "
            f"(accel_method={accel_method}, accel_level={accel_level})"
        )

    def _image_cb(self, msg: RosImage) -> None:
        """Receives an image, performs VLA inference, and publishes the action."""
        try:
            cv_img = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_img = Image.fromarray(cv_img)

            action = self._model.predict_action(
                image=pil_img,
                instruction=self._instruction,
                accel_method=self._accel_method,
                accel_config=self._method_config,
                prune_config=self._prune_config,
            )

            if action is not None:
                out_msg = Float64MultiArray()
                out_msg.data = [float(x) for x in action.flatten().tolist()]
                self._pub.publish(out_msg)
        except Exception:
            # Silence internal errors to keep the node running
            pass


def _ros_worker(
    model_name: str,
    model_path: str,
    input_topic: str,
    output_topic: str,
    accel_method: str,
    accel_level: str,
    instruction: str,
):
    """Multiprocessing worker function to initialize and run the VLA inference node."""
    rclpy.init()
    node = VLAServerNode(
        model_name,
        model_path,
        input_topic,
        output_topic,
        accel_method,
        accel_level,
        instruction=instruction
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ================= MCP Tools ================= #

@mcp.tool()
def start_vla_inference(
    model_name: str,
    model_path: Optional[str],   
    instruction: str,
    input_topic: str = "/isaac_rgb",
    output_topic: str = "/ee_command",
    accel_method: str = "none",
    accel_level: str = "off",
) -> str:
    """
    [VLA REASONING] Starts the Vision-Language-Action (VLA) node to execute complex tasks
    based on natural language instructions.

    The VLA node subscribes to visual input and **autonomously publishes a sequence of EE commands** to the control topic until the task is complete.

    REQUIRED TASK FLOW (To pick up an object):
    1. Call `start_camera` (to provide images to input_topic).
    2. Call `start_controller` (to make the robot listen on output_topic).
    3. Call this tool with the detailed 'instruction'.

    Args:
        model_name (str): The VLA model identifier (e.g., "openvla").
        model_path (str|None): Path to the model checkpoint. If None, attempt to read from config/vla_models.json.
        instruction (str): The natural language instruction defining the task (e.g., "pick up the blue bowl").
        input_topic (str): [Input Topic] Topic subscribed for images (must match camera output).
        output_topic (str): [Output Topic] Topic used to publish VLA-predicted actions (must match controller input).
        accel_method (str): Acceleration method name (e.g., "fastv").
        accel_level (str): Acceleration preset level.
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

    # If model_path not provided, try to load from config/vla_models.json
    if not model_path:
        try:
            models_cfg = _load_vla_models_config()
            model_path = models_cfg.get(model_name)
            if not model_path:
                return (
                    f"Error: model_path not provided and model '{model_name}' not found in config/vla_models.json."
                    f" Create config/vla_models.json with a mapping or pass --model-path."
                )
        except Exception as e:
            return f"Error: failed to load vla models config: {e}"

    # Validate acceleration method and level
    valid_methods = {"none", "fastv"}
    if accel_method not in valid_methods:
        return (
            f"Error: invalid accel_method '{accel_method}'. "
            f"Use one of {sorted(valid_methods)}."
        )

    valid_levels = {"off", "balanced", "aggressive"}
    if accel_method == "fastv" and accel_level not in valid_levels:
        return (
            f"Error: invalid accel_level '{accel_level}' for method 'fastv'. "
            f"Use one of {sorted(valid_levels)}."
        )

    # Use spawn to avoid fork-related rclpy issues
    ctx = multiprocessing.get_context("spawn")
    _VLA_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(model_name, model_path, input_topic, output_topic, accel_method, accel_level, instruction),
        daemon=False,
    )
    _VLA_PROCESS.start()
    return (
        f"Success: VLA {model_name} started (pid={_VLA_PROCESS.pid}) "
        f"with accel_method={accel_method}, accel_level={accel_level}."
    )


@mcp.tool()
def stop_vla_inference() -> str:
    """Stops the running VLA inference loop and cleans up the background process."""
    global _VLA_PROCESS
    if _VLA_PROCESS is None or not _VLA_PROCESS.is_alive():
        return "Info: No VLA is running."

    _VLA_PROCESS.terminate()
    _VLA_PROCESS.join(timeout=5.0)
    if _VLA_PROCESS.is_alive():
        try:
            _VLA_PROCESS.kill()
        except Exception:
            pass
        _VLA_PROCESS.join(timeout=1.0)

    _VLA_PROCESS = None
    return "Success: VLA stopped."


if __name__ == "__main__":
    # Local test harness
    import argparse
    import sys
    import select

    parser = argparse.ArgumentParser(description="vla_server.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--model-name", type=str, default="default_model", help="Model registry key to test")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--instruction", type=str, default="do task", help="Natural language instruction for the VLA task")
    parser.add_argument("--input-topic", type=str, default="/isaac_rgb", help="Input image topic")
    parser.add_argument("--output-topic", type=str, default="/ee_command", help="Output action topic")
    parser.add_argument("--accel-method", type=str, default="none", help="Acceleration method: none | fastv")
    parser.add_argument("--accel-level", type=str, default="off", help="Acceleration level for fastv: off | balanced | aggressive")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: attempting to start VLA worker (spawn)...")
        res = start_vla_inference(
            args.model_name,
            args.model_path,
            args.instruction,
            args.input_topic,
            args.output_topic,
            args.accel_method,
            args.accel_level,
        )
        print(res)
        if res.startswith("Error"):
            sys.exit(1)

        try:
            print("VLA worker started. Press Ctrl-C to stop, or type 'stop' + Enter.")
            while True:
                time.sleep(0.5)
                # Check process liveness
                if _VLA_PROCESS is None or not _VLA_PROCESS.is_alive():
                    print("VLA process exited.")
                    break
                # simple stdin check
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping VLA...")
        finally:
            print(stop_vla_inference())
            print("Local test finished.")
    else:
        # Run as an MCP service
        mcp.run()
