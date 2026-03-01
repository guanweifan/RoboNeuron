#!/usr/bin/env python3
"""ROS2 node: simulated_robot

Core functionality:
- Publishes simulated joint states to 'isaac_joint_states' (sensor_msgs/JointState).
- Subscribes to joint commands from 'isaac_joint_commands' (sensor_msgs/JointState).
- Supports different robot configurations (Panda/FR3) via ROS parameters.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image 
from typing import Dict, Any, List, Optional
import numpy as np

# --- Robot Configuration Data ---
ROBOT_CONFIG: Dict[str, Any] = {
    "panda": {
        "joint_names": [
            'panda_joint1','panda_joint2','panda_joint3','panda_joint4','panda_joint5',
            'panda_joint6','panda_joint7','panda_finger_joint1','panda_finger_joint2'
        ],
        "default_position": [0.012, -0.5686, 0.0, -2.8106, 0.0, 3.0367, 0.741, 0.0, 0.0],
        "default_effort": [0.0, -6.5535, 0.0, 18.6816, 0.0, 1.0512, 0.0, 0.0044, -0.0044]
    },
    "fr3": {
        "joint_names": [
            'fr3_joint1', 'fr3_joint2', 'fr3_joint3', 'fr3_joint4', 
            'fr3_joint5', 'fr3_joint6', 'fr3_joint7', 
            'fr3_finger_joint1', 'fr3_finger_joint2' 
        ],
        "default_position": [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.0, 0.0],
        "default_effort": [0.0] * 9 
    }
}

class SimulatedRobot(Node):
    """
    ROS2 node simulating a robot's joint state and control loop.

    It reads the 'robot_model' parameter to select configuration and uses a timer 
    to periodically publish joint states based on received commands.
    """

    def __init__(self):
        super().__init__('simulated_robot')

        # 1. Parameter Declaration and Retrieval
        self.declare_parameter('robot_model', 'fr3')
        self.robot_model = self.get_parameter('robot_model').get_parameter_value().string_value
        
        if self.robot_model not in ROBOT_CONFIG:
            self.get_logger().error(f"Unsupported robot model: {self.robot_model}. Defaulting to 'panda'.")
            self.robot_model = 'panda'

        self.config = ROBOT_CONFIG[self.robot_model]
        self.get_logger().info(f'Simulating robot: {self.robot_model.upper()}')

        # 2. State Variable Initialization
        self.joint_names: List[str] = self.config["joint_names"]
        self.num_joints: int = len(self.joint_names)
        
        self.joint_positions: List[float] = list(self.config["default_position"])
        self.joint_velocity: List[float] = [0.0] * self.num_joints
        self.joint_effort: List[float] = self.config["default_effort"]

        # 3. ROS 2 Interface Setup
        self.joint_states_pub = self.create_publisher(JointState, 'isaac_joint_states', 10)
        self.joint_commands_sub = self.create_subscription(
            JointState, 
            'isaac_joint_commands', 
            self.joint_command_cb, 
            10
        )

        # 4. Timer Setup
        self.seq: int = 0
        self.rate_hz: int = 100 
        self.create_timer(1.0 / self.rate_hz, self.publish_joint_states)

        self.get_logger().info('simulated_robot node started.')
        
    def joint_command_cb(self, msg: JointState) -> None:
        """
        Receives joint commands and updates the internal simulated joint positions.

        Args:
            msg (JointState): The incoming joint command message.
        """
        if not msg.name or not msg.position:
            self.get_logger().warn('Received empty JointState command.')
            return

        # Simple simulation: immediately update current position to commanded position
        for name, cmd_pos in zip(msg.name, msg.position):
            try:
                idx = self.joint_names.index(name)
                self.joint_positions[idx] = cmd_pos 
            except ValueError:
                self.get_logger().warn(f'Joint "{name}" not found in current model ({self.robot_model}).')

    def publish_joint_states(self) -> None:
        """Publishes the current simulated joint state."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        
        # Publish current state
        msg.position = self.joint_positions
        msg.velocity = self.joint_velocity
        msg.effort = self.joint_effort
        
        self.joint_states_pub.publish(msg)


def main(args=None) -> None:
    """Entry point for the ROS2 node."""
    rclpy.init(args=args)
    node = SimulatedRobot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('simulated_robot shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()