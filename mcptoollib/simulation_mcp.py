import cv2
import rclpy
import importlib
import threading
import multiprocessing
import sys
import time
import select
import argparse
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from mcp.server.fastmcp import FastMCP
from typing import Optional, Any, Tuple, Dict, List

# Global storage for the background process
_SIMULATION_PROCESS: Optional[multiprocessing.Process] = None

mcp = FastMCP("robomcp-simulation")

# --- Simulation ROS Node Logic ---

class SimulationPublishNode(Node):
    """
    ROS2 Node managing the simulation environment within a multiprocessing worker.

    Initializes the AdapterWrapper, runs the simulation environment, and 
    periodically publishes observations (images) while subscribing to commands.
    """
    def __init__(self, adapter_wrapper_cls: Any, suite: str, task_id: int, public_topic: str, input_topic: str, rate_hz: int):
        super().__init__('rgb_publisher_node')
        self.locker = threading.Lock()
        self._cv_bridge = CvBridge()
        
        self._adapter = adapter_wrapper_cls(suite, task_id)
        self._adapter.create_simulation_environment()
        
        self._timer = self.create_timer(1.0 / rate_hz, self._timer_callback)
        self.get_logger().info(f"Simulation started on {public_topic} at {rate_hz}Hz")

        self.timestamp: int = 0

        self._publisher = self.create_publisher(RosImage, public_topic, 10)
        self.create_subscription(Float64MultiArray, input_topic, self.robot_update_cb, 10)


    def _timer_callback(self) -> None:
        """
        Periodically obtains the latest observation and publishes the primary RGB image.
        """
        with self.locker:
            obs_results: Tuple[Dict[str, Any], Any, str] = self._adapter.obtain_observation()
            visual_observation: Dict[str, Any] = obs_results[0]
            
            if visual_observation is not None:
                # Publishes agentview_image, suitable for OpenVLA
                agentview_image = visual_observation["agentview_image"]
                ros_img_msg = self._cv_bridge.cv2_to_imgmsg(agentview_image, encoding='rgb8')
                ros_img_msg.header.stamp = self.get_clock().now().to_msg()
                self._publisher.publish(ros_img_msg)

    def robot_update_cb(self, msg: Float64MultiArray) -> None:
        """
        Callback to receive the action command and step the simulation.
        
        Args:
            msg (Float64MultiArray): The incoming action command vector.
        """
        with self.locker:
            action: List[float] = list(msg.data)
            # Action vector format assumption: [delta x y z r p y gripper]
            # Gripper range is expected to be [0, 1] (open < 0.5, close > 0.5)
            self._adapter.update_environment(action)
            
            # After stepping, save a frame for debugging
            observation, _, _ = self._adapter.obtain_observation()
            if observation is not None:
                cv2.imwrite(f"/simulation_output/{self.timestamp}.png", observation["agentview_image"])
                self.timestamp += 1

    def destroy_node(self) -> None:
        """Cleans up resources, including closing the simulation environment."""
        self._adapter.env.close()
        super().destroy_node()

def _ros_worker(wrapper_import_path: str, suite: str, task_id: int, public_topic: str, input_topic: str, rate_hz: int = 10) -> None:
    """Multiprocessing worker function to initialize and run the ROS node."""
    module_name, class_name = wrapper_import_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    wrapper_cls = getattr(mod, class_name)

    rclpy.init()
    node = SimulationPublishNode(wrapper_cls, suite, task_id, public_topic, input_topic, rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

@mcp.tool()
def start_simulation(wrapper_import: str, suite: str, task_id: int, public_topic: str = "/simulation_rgb", input_topic: str = "/ee_command") -> str:
    """
    Starts a ROS node publishing simulation images in the background using multiprocessing.

    Args:
        wrapper_import (str): The import path of the AdapterWrapper class.
        suite (str): The name of the task suite.
        task_id (int): The ID of the task within the suite.
        public_topic (str): The topic for publishing simulation images.
        input_topic (str): The topic for receiving robot commands.
    """
    global _SIMULATION_PROCESS
    if _SIMULATION_PROCESS is not None and _SIMULATION_PROCESS.is_alive():
        return "Error: Simulation process is already running."

    # Use 'spawn' for safe multiprocessing with rclpy/ROS
    ctx = multiprocessing.get_context('spawn')
    try:
        module_name, class_name = wrapper_import.rsplit(".", 1)
        importlib.import_module(module_name)
    except Exception as e:
        return f"Error: failed to import adapter wrapper '{wrapper_import}': {e}"
    
    _SIMULATION_PROCESS = ctx.Process(
        target = _ros_worker,
        args = (wrapper_import, suite, task_id, public_topic, input_topic),
        daemon = False
    )
    _SIMULATION_PROCESS.start()
    return f"Success: Simulation started publishing to {public_topic} (pid={_SIMULATION_PROCESS.pid})."

@mcp.tool()
def stop_simulation() -> str:
    """Stops the currently running simulation ROS node and cleans up the process."""
    global _SIMULATION_PROCESS
    if _SIMULATION_PROCESS is None or not _SIMULATION_PROCESS.is_alive():
        return "Info: No simulation process is running."
    
    _SIMULATION_PROCESS.terminate()
    _SIMULATION_PROCESS.join(timeout=5.0)
    if _SIMULATION_PROCESS.is_alive():
        try:
            _SIMULATION_PROCESS.kill()
        except Exception:
            pass
        _SIMULATION_PROCESS.join(timeout=1.0)
    _SIMULATION_PROCESS = None
    return "Success: Simulation process stopped."


class ControlNode:
    """
    ROS2 node simulating a minimal policy control loop for local testing.

    Subscribes to simulation images and publishes a constant dummy action.
    """
    def __init__(self, input_topic: str = "/ee_command", publish_topic: str = "/ee_command"):
        self.node = rclpy.create_node("control_node")
        self._cv_bridge = CvBridge()
        
        self.publisher = self.node.create_publisher(Float64MultiArray, publish_topic, 10)
        self.subscription = self.node.create_subscription(
            RosImage, "/simulation_rgb", self.image_callback, 10
        )
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
        action_msg = Float64MultiArray()
        action_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # demo action
        self.publisher.publish(action_msg)

    def stop(self) -> None:
        """Cleans up the control node and stops the spinning thread."""
        self.running = False
        self.node.destroy_node()
        rclpy.shutdown()
        self.thread.join()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Simulation local test harness")
    # Using default=True based on original code structure
    parser.add_argument("--local-test", default=True, help="Run local simulation test instead of MCP server")
    parser.add_argument("--wrapper", type=str, default="wrapper.robot_wrapper.libero_adapter.LiberoAdapterWrapper", help="Adapter wrapper import path")
    parser.add_argument("--suite", type=str, default="libero_spatial", help="Task suite name")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: starting simulation node...")
        res = start_simulation(args.wrapper, args.suite, args.task_id)
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
        # Run as an MCP service
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("robomcp-simulation")
        mcp.run()