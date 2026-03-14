import time
from typing import Any

import rclpy
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from rclpy.node import Node
from roboneuron_interfaces.msg import EEFDeltaCommand

from roboneuron_core.utils.eef_delta import EEF_DELTA_CMD_TOPIC

mcp = FastMCP("eef-delta-mcptool")


class EEFDeltaInput(BaseModel):
    delta_x: float
    delta_y: float
    delta_z: float
    delta_roll: float
    delta_pitch: float
    delta_yaw: float
    gripper_cmd: float


def populate_ros_message(ros_msg, data_dict: dict[str, Any]):
    """Populate a ROS 2 message object from a Pydantic dictionary."""
    for key, value in data_dict.items():
        if not hasattr(ros_msg, key):
            continue

        attr = getattr(ros_msg, key)

        if hasattr(attr, "get_fields_and_field_types"):
            if isinstance(value, dict):
                populate_ros_message(attr, value)
        else:
            setattr(ros_msg, key, value)


class EEFDeltaPublisher(Node):
    """Native ROS 2 node that publishes ``EEFDeltaCommand`` messages."""

    def __init__(self):
        super().__init__("mcp_eef_delta_publisher")
        self.publisher_ = self.create_publisher(EEFDeltaCommand, EEF_DELTA_CMD_TOPIC, 10)
        self.get_logger().info(f"MCP tool node initialized for topic: {EEF_DELTA_CMD_TOPIC}")

    def publish(self, data: EEFDeltaInput) -> dict[str, Any]:
        """Convert the Pydantic payload to ROS and publish it."""
        ros_msg = EEFDeltaCommand()
        data_dict = data.model_dump()
        populate_ros_message(ros_msg, data_dict)
        self.publisher_.publish(ros_msg)
        return {
            "op": "publish",
            "topic": EEF_DELTA_CMD_TOPIC,
            "timestamp": time.time(),
            "data": data_dict,
        }

    def publish_seq(
        self,
        data_seq: list[EEFDeltaInput],
        duration_seq: list[float],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for data, duration in zip(data_seq, duration_seq, strict=False):
            result = self.publish(data)
            results.append(result)
            time.sleep(duration)
        return results


ros_node: EEFDeltaPublisher | None = None


def init_ros_node() -> None:
    global ros_node
    try:
        if not rclpy.ok():
            rclpy.init()
        ros_node = EEFDeltaPublisher()
    except Exception as e:
        print(f"Failed to initialize ROS node: {e}")


init_ros_node()


@mcp.tool()
def pub_eef_delta(data: EEFDeltaInput):
    """Publish a single ``EEFDeltaCommand`` message to ``/eef_delta_cmd``."""
    if ros_node is None:
        return "ROS Node not initialized."

    try:
        ros_node.publish(data)
        return f"Published to {EEF_DELTA_CMD_TOPIC}"
    except Exception as e:
        return f"Failed to publish: {e}"


@mcp.tool()
def pub_eef_delta_seq(data_seq: list[EEFDeltaInput], duration_seq: list[float]):
    """Publish a sequence of ``EEFDeltaCommand`` messages."""
    if ros_node is None:
        return "ROS Node not initialized."

    try:
        ros_node.publish_seq(data_seq, duration_seq)
        return "Sequence published successfully"
    except Exception as e:
        return f"Sequence failed: {e}"

if __name__ == "__main__":
    print("Starting MCP server for EEFDeltaCommand...")
    mcp.run(transport="stdio")
