import argparse
import importlib
import multiprocessing
import os
import select
import sys
import threading
import time
from contextlib import suppress
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from mcp.server.fastmcp import FastMCP
from rclpy.node import Node
from roboneuron_interfaces.msg import EEFDeltaCommand
from sensor_msgs.msg import Image as RosImage

from roboneuron_core.utils.eef_delta import EEF_DELTA_CMD_TOPIC, array_to_eef_delta_command, eef_delta_command_to_array

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Global storage for the background process
_SIMULATION_PROCESS: multiprocessing.Process | None = None

mcp = FastMCP("robomcp-simulation")

# --- Simulation ROS Node Logic ---

def _extract_visual_from_obtain(obtain_ret):
    # old libero: (visual_dict, ..., ...)
    if isinstance(obtain_ret, tuple) and len(obtain_ret) > 0 and isinstance(obtain_ret[0], dict):
        return obtain_ret[0]

    # new standardized: {"visual_observation": {...}, ...}
    if isinstance(obtain_ret, dict):
        vo = obtain_ret.get("visual_observation")
        if isinstance(vo, dict):
            return vo
        # sometimes obtain_observation directly returns visual dict
        return obtain_ret

    return {}


def _resolve_adapter_kwargs(
    suite: Any | None,
    task_id: int | None,
    adapter_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Resolve adapter kwargs while preserving old and new start_simulation call patterns.
    """
    if adapter_config is not None:
        if not isinstance(adapter_config, dict):
            raise TypeError("adapter_config must be a dictionary.")
        return dict(adapter_config)

    # Compatibility with newer callers that pass adapter_config as the second positional arg.
    if isinstance(suite, dict):
        return dict(suite)

    # Legacy signature compatibility: start_simulation(wrapper_import, suite, task_id, ...)
    return {
        "libero_suite_name": suite if isinstance(suite, str) else "libero_spatial",
        "task_id": 0 if task_id is None else int(task_id),
    }


class SimulationPublishNode(Node):
    """
    ROS2 Node managing the simulation environment within a multiprocessing worker.

    Initializes the AdapterWrapper, runs the simulation environment, and
    periodically publishes observations (images) while subscribing to commands.
    """
    def __init__(self, adapter_wrapper_cls: Any, adapter_kwargs: dict[str, Any], public_topic: str, input_topic: str, rate_hz: int):
        super().__init__('rgb_publisher_node')
        self.locker = threading.Lock()
        self._cv_bridge = CvBridge()

        # Initialize adapter with flexible kwargs
        self._adapter = adapter_wrapper_cls(**adapter_kwargs)
        # Note: Adapter's __init__ already calls create_simulation_environment() if available
        # Only call it here if the adapter doesn't have an initialized environment
        if not hasattr(self._adapter, 'env') or self._adapter.env is None:
            self._adapter.create_simulation_environment()

        self._timer = self.create_timer(1.0 / rate_hz, self._timer_callback)
        self.get_logger().info(f"Simulation started on {public_topic} at {rate_hz}Hz")

        self.timestamp: int = 0
# --- Debug frame output dir ---
        self.output_dir = "/tmp/simulation_output"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            self.get_logger().warning(f"Failed to create output dir {self.output_dir}: {e}")
        self._publisher = self.create_publisher(RosImage, public_topic, 10)
        self.create_subscription(EEFDeltaCommand, input_topic, self.robot_update_cb, 10)


    def _timer_callback(self) -> None:
        """
        Periodically obtains the latest observation and publishes the primary RGB image.
        """
        with self.locker:
            obtain_ret = self._adapter.obtain_observation()
            visual_observation = _extract_visual_from_obtain(obtain_ret)


            if visual_observation:
                # Try different possible image keys (prioritize agentview, fallback to others)
                image_keys = ["agentview_image", "static_rgb", "rgb_static"]
                selected_image = None

                for key in image_keys:
                    if key in visual_observation:
                        selected_image = visual_observation[key]
                        self.get_logger().debug(f"Publishing image from key: {key}")
                        break

                if selected_image is not None:
                    # Convert numpy array to ROS Image message
                    if isinstance(selected_image, np.ndarray):
                        ros_img_msg = self._cv_bridge.cv2_to_imgmsg(selected_image, encoding='rgb8')
                        ros_img_msg.header.stamp = self.get_clock().now().to_msg()
                        self._publisher.publish(ros_img_msg)
                    else:
                        self.get_logger().warning(f"Image is not a numpy array: {type(selected_image)}")
                else:
                    self.get_logger().warning("No suitable image found in visual_observation")

    def robot_update_cb(self, msg: EEFDeltaCommand) -> None:
        with self.locker:
            action = eef_delta_command_to_array(msg).tolist()

            # 1) Step/update env (compatible with libero / calvin)
            try:
                if hasattr(self._adapter, "update_environment"):
                    self._adapter.update_environment(action)
                elif hasattr(self._adapter, "step"):
                    self._adapter.step(action)
                else:
                    self.get_logger().error("Adapter has neither update_environment nor step.")
                    return
            except Exception as e:
                self.get_logger().error(f"Failed to step/update environment: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                return

            # 2) Always obtain fresh observation for saving frames (tuple/dict compatible)
            try:
                obtain_ret = self._adapter.obtain_observation()
                visual_obs = _extract_visual_from_obtain(obtain_ret)
            except Exception as e:
                self.get_logger().error(f"Failed to obtain observation after step: {e}")
                import traceback
                self.get_logger().error(traceback.format_exc())
                return

            # 3) Save frame if any
            image_keys = ["agentview_image", "static_rgb", "rgb_static"]
            for key in image_keys:
                if key in visual_obs and isinstance(visual_obs[key], np.ndarray):
                    img = visual_obs[key]
                    try:
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    except Exception:
                        img_bgr = img
                    out_path = os.path.join(getattr(self, "output_dir", "/simulation_output"),
                                            f"{self.timestamp}.png")
                    ok = cv2.imwrite(out_path, img_bgr)
                    if ok:
                        self.timestamp += 1
                    else:
                        self.get_logger().warning(f"cv2.imwrite failed: {out_path}")
                    break

    def destroy_node(self) -> None:
        """Cleans up resources, including closing the simulation environment."""
        # Try to close environment gracefully if it exists
        try:
            if (
                hasattr(self._adapter, "env")
                and self._adapter.env is not None
                and hasattr(self._adapter.env, "close")
            ):
                self._adapter.env.close()
        except Exception as e:
            self.get_logger().warning(f"Failed to close environment: {e}")

        super().destroy_node()

def _ros_worker(wrapper_import_path: str, adapter_kwargs: dict[str, Any], public_topic: str, input_topic: str, rate_hz: int = 10) -> None:
    import time
    import traceback
    log_path = "/tmp/robomcp_sim_worker.log"
    try:
        module_name, class_name = wrapper_import_path.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        wrapper_cls = getattr(mod, class_name)

        rclpy.init()
        node = SimulationPublishNode(wrapper_cls, adapter_kwargs, public_topic, input_topic, rate_hz)
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        finally:
            node.destroy_node()
            rclpy.shutdown()

    except Exception:
        with open(log_path, "a") as f:
            f.write("\n" + "="*80 + "\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] worker crashed\n")
            f.write(f"wrapper_import_path={wrapper_import_path}\n")
            f.write(f"adapter_kwargs={adapter_kwargs}\n")
            f.write(traceback.format_exc())
        print(f"[worker] crashed, see {log_path}")
        raise


@mcp.tool()
def start_simulation(
    wrapper_import: str,
    suite: Any | None = None,
    task_id: int | None = None,
    public_topic: str = "/simulation_rgb",
    input_topic: str = EEF_DELTA_CMD_TOPIC,
    adapter_config: dict[str, Any] | None = None,
    rate_hz: int = 10,
) -> str:
    """
    Starts a ROS node publishing simulation images in the background using multiprocessing.

    Args:
        wrapper_import (str): The import path of the AdapterWrapper class (e.g., "roboneuron_core.adapters.robot.libero_adapter.LiberoAdapterWrapper").
        suite (str|dict|None): Legacy suite name, or adapter_config dict when passed positionally.
        task_id (int|None): Legacy task id for old callers.
        public_topic (str): The topic for publishing simulation images.
        input_topic (str): The topic for receiving robot commands.
        adapter_config (Dict[str, Any]|None): Explicit adapter kwargs for flexible adapters.
        rate_hz (int): Publishing rate in Hz.
    """
    global _SIMULATION_PROCESS
    if _SIMULATION_PROCESS is not None and _SIMULATION_PROCESS.is_alive():
        return "Error: Simulation process is already running."

    try:
        resolved_adapter_kwargs = _resolve_adapter_kwargs(suite, task_id, adapter_config)
    except (TypeError, ValueError) as exc:
        return f"Error: {exc}"

    # Use 'spawn' for safe multiprocessing with rclpy/ROS
    ctx = multiprocessing.get_context('spawn')
    try:
        module_name, class_name = wrapper_import.rsplit(".", 1)
        importlib.import_module(module_name)
    except Exception as e:
        return f"Error: failed to import adapter wrapper '{wrapper_import}': {e}"

    _SIMULATION_PROCESS = ctx.Process(
        target = _ros_worker,
        args = (wrapper_import, resolved_adapter_kwargs, public_topic, input_topic, rate_hz),
        daemon = False
    )
    _SIMULATION_PROCESS.start()
    return f"Success: Simulation started publishing to {public_topic} at {rate_hz}Hz (pid={_SIMULATION_PROCESS.pid})."

@mcp.tool()
def stop_simulation() -> str:
    """Stops the currently running simulation ROS node and cleans up the process."""
    global _SIMULATION_PROCESS
    if _SIMULATION_PROCESS is None or not _SIMULATION_PROCESS.is_alive():
        return "Info: No simulation process is running."
    
    _SIMULATION_PROCESS.terminate()
    _SIMULATION_PROCESS.join(timeout=5.0)
    if _SIMULATION_PROCESS.is_alive():
        with suppress(Exception):
            _SIMULATION_PROCESS.kill()
        _SIMULATION_PROCESS.join(timeout=1.0)
    _SIMULATION_PROCESS = None
    return "Success: Simulation process stopped."


class ControlNode:
    """
    ROS2 node simulating a minimal policy control loop for local testing.

    Subscribes to simulation images and publishes a constant dummy EEF delta command.
    """
    def __init__(self, input_topic: str = EEF_DELTA_CMD_TOPIC, publish_topic: str = EEF_DELTA_CMD_TOPIC):
        self.node = rclpy.create_node("control_node")
        self._cv_bridge = CvBridge()

        self.publisher = self.node.create_publisher(EEFDeltaCommand, publish_topic, 10)
        self.subscription = self.node.create_subscription(RosImage, "/simulation_rgb", self.image_callback, 10)
        self.latest_image = None
        
        # Run ROS spinning in a separate thread
        self.running = True
        self.thread = threading.Thread(target=self.spin_thread, daemon=True)
        self.thread.start()

    def spin_thread(self) -> None:
        """The target function for the background thread to run the ROS 2 event loop."""
        rclpy.spin(self.node)

    def image_callback(self, msg: RosImage) -> None:
        """
        Receives an image, converts it, and publishes a dummy action.
        
        Args:
            msg (RosImage): The incoming image message.
        """
        cv_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_image = cv_image
        
        # Policy Logic Placeholder: VLA model inference would occur here
        action_msg = array_to_eef_delta_command([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.publisher.publish(action_msg)

    def stop(self) -> None:
        """Cleans up the control node and stops the spinning thread."""
        self.running = False
        self.node.destroy_node()
        rclpy.shutdown()
        self.thread.join()


def create_adapter_config_from_args(wrapper_name: str, **kwargs) -> dict[str, Any]:
    """Create adapter configuration based on wrapper type."""
    if wrapper_name == "dummy":
        return {
            "image_size": kwargs.get("resolution", 128),
            "instruction": kwargs.get("instruction", "execute dummy task"),
            "action_scale": kwargs.get("action_scale", 0.1),
        }
    if wrapper_name == "libero":
        return {
            "libero_suite_name": kwargs.get("suite", "libero_spatial"),
            "task_id": kwargs.get("task_id", 0),
            "visual_resolution": kwargs.get("resolution", 768),
            "step_substeps": kwargs.get("substeps", 10)
        }
    elif wrapper_name == "calvin":
        return {
            "dataset_path": kwargs.get("dataset_path", ""),
            "task_id": kwargs.get("task_id", 0),
            "show_gui": kwargs.get("show_gui", False),
            "visual_resolution": kwargs.get("resolution", 200),
            "step_substeps": kwargs.get("substeps", 1),
            "load_sequences": kwargs.get("load_sequences", False)
        }
    else:
        # Default configuration - try to map common parameters
        return {
            "task_id": kwargs.get("task_id", 0),
            **kwargs
        }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Simulation local test harness")
    parser.add_argument("--local-test", action="store_true", default=False, help="Run local simulation test instead of MCP server")
    parser.add_argument("--wrapper", type=str, default="dummy", help="Adapter wrapper name (dummy, libero, calvin) or import path")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="Task suite name (for libero)")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID")
    parser.add_argument("--dataset-path", type=str, help="Dataset path (for calvin)")
    parser.add_argument("--show-gui", action="store_true", default=False, help="Show GUI (for calvin)")
    parser.add_argument("--resolution", type=int, default=768, help="Visual resolution")
    parser.add_argument("--substeps", type=int, default=10, help="Step substeps")
    parser.add_argument("--instruction", type=str, default="execute dummy task", help="Task instruction (for dummy adapter)")
    parser.add_argument("--action-scale", type=float, default=0.1, help="Action scale (for dummy adapter)")
    parser.add_argument("--rate-hz", type=int, default=10, help="Publishing rate in Hz")
    parser.add_argument("--load-sequences", action="store_true", default=False,
                    help="(calvin) load evaluation sequences (slow to start, only for calvin)")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: starting simulation node...")

        # Determine wrapper import path and config
        if args.wrapper in ["dummy", "libero", "calvin"]:
            from roboneuron_core.adapters.robot import get_registry
            registry = get_registry()
            if args.wrapper not in registry:
                print(f"Error: Adapter '{args.wrapper}' not found in registry")
                sys.exit(1)
            wrapper_cls = registry[args.wrapper]
            wrapper_import = f"{wrapper_cls.__module__}.{wrapper_cls.__name__}"
        else:
            # Assume it's a full import path
            wrapper_import = args.wrapper

        # Create adapter configuration
        adapter_config = create_adapter_config_from_args(
            args.wrapper,
            suite=args.suite,
            task_id=args.task_id,
            dataset_path=args.dataset_path,
            show_gui=args.show_gui,
            resolution=args.resolution,
            substeps=args.substeps,
            instruction=args.instruction,
            action_scale=args.action_scale,
            load_sequences=args.load_sequences, 
        )

        print(f"Using wrapper: {wrapper_import}")
        print(f"Adapter config: {adapter_config}")

        res = start_simulation(wrapper_import, adapter_config=adapter_config, rate_hz=args.rate_hz)
        print(res)
        if res.startswith("Error"):
            sys.exit(1)

        print("Starting control node...")
        rclpy.init()
        control_node = ControlNode()

        try:
            print("Simulation running. Press Ctrl-C to stop, or type 'stop' + Enter.")
            while True:
                time.sleep(0.5)
                # check stdin for stop command
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping simulation...")
        finally:
            control_node.stop()
            print(stop_simulation())
            print("Local simulation test finished.")
    else:
        mcp.run()
