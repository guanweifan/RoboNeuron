from typing import List, Dict, Any
from pydantic import BaseModel
import time
import threading

# ROS 2 Imports
import rclpy
from rclpy.node import Node
# Dynamic import based on generator context
from geometry_msgs.msg import Twist

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP
mcp = FastMCP("twist-mcptool")

# --- Pydantic Models Generation ---
class Linear(BaseModel):
    x: float
    y: float
    z: float

class Angular(BaseModel):
    x: float
    y: float
    z: float

class TwistInput(BaseModel):
    linear: Linear
    angular: Angular


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
class TwistPublisher(Node):
    """
    Native ROS 2 Node to publish Twist messages.
    """
    def __init__(self):
        super().__init__('mcp_twist_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('MCP Tool Node initialized for topic: /cmd_vel')

    def publish(self, data: TwistInput) -> Dict:
        """
        Converts Pydantic model to ROS message and publishes it.
        """
        ros_msg = Twist()
        
        # Convert Pydantic model to dict, then populate ROS message
        data_dict = data.model_dump()
        populate_ros_message(ros_msg, data_dict)

        self.publisher_.publish(ros_msg)
        
        return {
            "op": "publish",
            "topic": '/cmd_vel',
            "timestamp": time.time(),
            "data": data_dict
        }

    def publish_seq(self, data_seq: List[TwistInput], duration_seq: List[float]) -> List[Dict]:
        results = []
        for data, duration in zip(data_seq, duration_seq):
            result = self.publish(data)
            results.append(result)
            time.sleep(duration)
        return results


# --- Global ROS 2 Initialization ---
# MCP tools are stateless functions, so we need a global node instance.
ros_node: TwistPublisher = None

def init_ros_node():
    global ros_node
    try:
        # Check if rclpy is already initialized (e.g. by another tool)
        if not rclpy.ok():
            rclpy.init()
        ros_node = TwistPublisher()
        
        # Optional: Spin in a separate thread if you need subscriptions later
        # thread = threading.Thread(target=rclpy.spin, args=(ros_node,), daemon=True)
        # thread.start()
        
    except Exception as e:
        print(f"Failed to initialize ROS node: {e}")

# Initialize immediately upon module load
init_ros_node()


# --- MCP Tools ---

@mcp.tool()
def pub_twist(data: TwistInput):
    """
    Publishes a single Twist message to /cmd_vel.
    """
    if not ros_node:
        return "ROS Node not initialized."
    
    try:
        result = ros_node.publish(data)
        return f"Published to /cmd_vel"
    except Exception as e:
        return f"Failed to publish: {e}"

@mcp.tool()
def pub_twist_seq(data_seq: List[TwistInput], duration_seq: List[float]):
    """
    Publishes a sequence of Twist messages.
    """
    if not ros_node:
        return "ROS Node not initialized."

    try:
        ros_node.publish_seq(data_seq, duration_seq)
        return "Sequence published successfully"
    except Exception as e:
        return f"Sequence failed: {e}"

if __name__ == "__main__":
    print(f"Starting MCP Server for Twist...")
    mcp.run(transport="stdio")