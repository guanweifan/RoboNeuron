from typing import List, Dict, Any
from pydantic import BaseModel
import time
import threading

# ROS 2 Imports
import rclpy
from rclpy.node import Node
# Dynamic import based on generator context
from diy_msgs.msg import EECommand

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("eecommand-mcptool")

# --- Pydantic Models Generation ---
class EECommandInput(BaseModel):
    delta_x: float
    delta_y: float
    delta_z: float
    delta_roll: float
    delta_pitch: float
    delta_yaw: float
    gripper_cmd: float


# --- Helper Function for ROS Message Population ---
def populate_ros_message(ros_msg, data_dict: Dict[str, Any]):
    """
    Recursively populates a ROS 2 message object from a dictionary (Pydantic dump).
    Handles nested ROS messages.
    """
    for key, value in data_dict.items():
        if not hasattr(ros_msg, key):
            continue
            
        attr = getattr(ros_msg, key)
        
        # Check if the attribute is a nested ROS message (has standard slots)
        if hasattr(attr, 'get_fields_and_field_types'):
            if isinstance(value, dict):
                populate_ros_message(attr, value)
        else:
            # Primitive type assignment
            setattr(ros_msg, key, value)


# --- ROS 2 Node Wrapper ---
class EECommandPublisher(Node):
    """
    Native ROS 2 Node to publish EECommand messages.
    """
    def __init__(self):
        super().__init__('mcp_eecommand_publisher')
        self.publisher_ = self.create_publisher(EECommand, 'ee_command', 10)
        self.get_logger().info('MCP Tool Node initialized for topic: ee_command')

    def publish(self, data: EECommandInput) -> Dict:
        """
        Converts Pydantic model to ROS message and publishes it.
        """
        ros_msg = EECommand()
        
        # Convert Pydantic model to dict, then populate ROS message
        data_dict = data.model_dump()
        populate_ros_message(ros_msg, data_dict)

        self.publisher_.publish(ros_msg)
        
        return {
            "op": "publish",
            "topic": 'ee_command',
            "timestamp": time.time(),
            "data": data_dict
        }

    def publish_seq(self, data_seq: List[EECommandInput], duration_seq: List[float]) -> List[Dict]:
        results = []
        for data, duration in zip(data_seq, duration_seq):
            result = self.publish(data)
            results.append(result)
            time.sleep(duration)
        return results


# --- Global ROS 2 Initialization ---
# MCP tools are stateless functions, so we need a global node instance.
ros_node: EECommandPublisher = None

def init_ros_node():
    global ros_node
    try:
        # Check if rclpy is already initialized (e.g. by another tool)
        if not rclpy.ok():
            rclpy.init()
        ros_node = EECommandPublisher()
        
        # Optional: Spin in a separate thread if you need subscriptions later
        # thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
        # thread.start()
        
    except Exception as e:
        print(f"Failed to initialize ROS node: {e}")

# Initialize immediately upon module load
init_ros_node()


# --- MCP Tools ---

@mcp.tool()
def pub_eecommand(data: EECommandInput):
    """
    Publishes a single EECommand message to ee_command.
    """
    if not ros_node:
        return "ROS Node not initialized."
    
    try:
        result = ros_node.publish(data)
        return f"Published to ee_command"
    except Exception as e:
        return f"Failed to publish: {e}"

@mcp.tool()
def pub_eecommand_seq(data_seq: List[EECommandInput], duration_seq: List[float]):
    """
    Publishes a sequence of EECommand messages.
    """
    if not ros_node:
        return "ROS Node not initialized."

    try:
        ros_node.publish_seq(data_seq, duration_seq)
        return "Sequence published successfully"
    except Exception as e:
        return f"Sequence failed: {e}"

if __name__ == "__main__":
    print(f"Starting MCP Server for EECommand...")
    mcp.run(transport="stdio")