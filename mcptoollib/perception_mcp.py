#!/usr/bin/env python3
"""
perception_mcp.py

MCP Server for managing RGB Camera ROS nodes.
Allows the LLM to start/stop camera streams on specific topics dynamically.
"""

import multiprocessing
import time
import importlib
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge
from mcp.server.fastmcp import FastMCP
import sys
import argparse
import select

# Global storage for the background process
_CAMERA_PROCESS = None

mcp = FastMCP("robomcp-camera")

# --- Original ROS Logic (Encapsulated) ---

class RGBPublisherNode(Node):
    """Original ROS2 Node logic for publishing images."""
    def __init__(self, camera_wrapper_cls, width, height, topic, rate_hz):
        super().__init__('rgb_publisher_node')
        self._publisher = self.create_publisher(RosImage, topic, 10)
        self._cv_bridge = CvBridge()
        self._camera = camera_wrapper_cls(width, height)
        self._camera.open()
        self._timer = self.create_timer(1.0 / rate_hz, self._timer_callback)
        self.get_logger().info(f"Camera started on {topic} at {rate_hz}Hz")

    def _timer_callback(self):
        ret, frame = self._camera.read()
        if ret and frame is not None:
            ros_msg = self._cv_bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            ros_msg.header.stamp = self.get_clock().now().to_msg()
            self._publisher.publish(ros_msg)

    def destroy_node(self):
        if self._camera:
            self._camera.close()
        super().destroy_node()

def _ros_worker(wrapper_import_path: str, topic: str, width: int, height: int):
    """Worker function to run the ROS node in a separate process."""
    # Dynamic import inside the process
    module_name, class_name = wrapper_import_path.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    wrapper_cls = getattr(mod, class_name)

    rclpy.init()
    node = RGBPublisherNode(wrapper_cls, width, height, topic, rate_hz=10)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

# --- MCP Tools ---

@mcp.tool()
def start_camera(wrapper_import: str, topic: str = "/isaac_rgb") -> str:
    """
    [PERCEPTION] Starts the camera node to stream visual input for VLA models.

    This is the first step in any task requiring visual feedback (e.g., picking, placing). 
    It streams RGB images to a specified ROS topic.
    
    Args:
        wrapper_import: Dot-path to the camera wrapper (e.g., 'wrapper.camera_wrapper.DummyCameraWrapper').
        topic: [Output Topic] The ROS topic used to publish the streaming RGB images (default: /isaac_rgb).
    """
    # ... (function body unchanged)
    global _CAMERA_PROCESS
    if _CAMERA_PROCESS is not None and _CAMERA_PROCESS.is_alive():
        return "Error: Camera is already running. Call stop_camera first."

    # Use 'spawn' start method to avoid fork-related issues with rclpy/ROS.
    ctx = multiprocessing.get_context('spawn')
    try:
        # Validate import early to give faster feedback
        module_name, class_name = wrapper_import.rsplit(".", 1)
        importlib.import_module(module_name)
    except Exception as e:
        return f"Error: failed to import camera wrapper '{wrapper_import}': {e}"

    _CAMERA_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(wrapper_import, topic, 256, 256),
        daemon=False  # do not mark as daemon so it survives parent thread handling during debugging
    )
    _CAMERA_PROCESS.start()
    return f"Success: Camera started publishing to {topic} (pid={_CAMERA_PROCESS.pid})."

@mcp.tool()
def stop_camera() -> str:
    """Stops the currently running camera ROS node."""
    global _CAMERA_PROCESS
    if _CAMERA_PROCESS is None or not _CAMERA_PROCESS.is_alive():
        return "Info: No camera is running."

    _CAMERA_PROCESS.terminate()
    _CAMERA_PROCESS.join(timeout=5.0)
    if _CAMERA_PROCESS.is_alive():
        # force kill (platform dependent); try again politely then detach
        try:
            _CAMERA_PROCESS.kill()
        except Exception:
            pass
        _CAMERA_PROCESS.join(timeout=1.0)

    _CAMERA_PROCESS = None
    return "Success: Camera stopped."

if __name__ == "__main__":
    # Provide a convenient local test harness so you can debug without running the MCP service.
    parser = argparse.ArgumentParser(description="rgb_test.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--wrapper", type=str, default="wrapper.camera_wrapper.DummyCameraWrapper",
                        help="Wrapper import path to test, e.g. 'wrapper.camera_wrapper.DummyCameraWrapper'")
    parser.add_argument("--topic", type=str, default="/isaac_rgb",
                        help="Topic to publish to during local test")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: attempting to start camera process with 'spawn'...")
        res = start_camera(args.wrapper, args.topic)
        print(res)
        if "Error" in res:
            print("Local test aborted due to error.")
            sys.exit(1)

        try:
            print("Camera process started. Press Ctrl-C to stop, or type 'stop' + Enter.")
            while True:
                try:
                    # non-blocking wait; let user type 'stop'
                    time.sleep(0.5)
                    if _CAMERA_PROCESS is None:
                        print("Camera process went away.")
                        break
                except KeyboardInterrupt:
                    print("\nKeyboardInterrupt received, stopping camera...")
                    break

                # simple stdin check (non-blocking is complex cross-platform; use small blocking read if available)
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
        except Exception as e:
            print(f"Exception in local test loop: {e}")
        finally:
            print("Stopping camera...")
            print(stop_camera())
            print("Local test finished.")
    else:
        # Normal behavior: run MCP server (no change)
        mcp.run()
