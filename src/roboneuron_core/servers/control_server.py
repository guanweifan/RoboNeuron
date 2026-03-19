#!/usr/bin/env python3
"""
RoboNeuron control server.

This MCP service hosts the ROS-side control runtime that resolves incoming
actions into robot command messages.
"""

from __future__ import annotations

import multiprocessing
import time
from contextlib import suppress

import rclpy
from mcp.server.fastmcp import FastMCP
from rclpy.node import Node
from roboneuron_interfaces.msg import EEFDeltaCommand, RawActionChunk
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboneuron_core.utils.control_runtime import (
    ActionChunk,
    ControlRuntime,
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    NormalizedCartesianVelocityConfig,
    RawActionStep,
    URDFKinematicsResolver,
)
from roboneuron_core.utils.eef_delta import EEF_DELTA_CMD_TOPIC, eef_delta_command_to_array
from roboneuron_core.utils.raw_action_chunk import (
    RAW_ACTION_CHUNK_TOPIC,
    raw_action_chunk_message_to_action_chunk,
)

_CONTROL_PROCESS = None
mcp = FastMCP("roboneuron-control")


class ControlRuntimeNode(Node):
    """ROS 2 control host that bridges action protocols to robot actuation."""

    def __init__(
        self,
        urdf_path: str,
        cartesian_cmd_topic: str,
        state_feedback_topic: str,
        joint_cmd_topic: str,
        cmd_msg_type: str = "JointState",
        raw_action_topic: str = RAW_ACTION_CHUNK_TOPIC,
        raw_action_protocol: str = DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
        raw_action_frame: str = "tool",
        max_linear_delta: float = 0.075,
        max_rotation_delta: float = 0.15,
        invert_gripper_action: bool = False,
    ) -> None:
        super().__init__("control_runtime_node")

        self.cmd_msg_type = cmd_msg_type
        self.current_joints: dict[str, float] = {}
        self._default_raw_action_protocol = raw_action_protocol
        self._default_raw_action_frame = raw_action_frame

        normalized_velocity_config = NormalizedCartesianVelocityConfig(
            max_linear_delta=max_linear_delta,
            max_rotation_delta=max_rotation_delta,
            frame=raw_action_frame,
            invert_gripper=invert_gripper_action,
        )
        resolver = URDFKinematicsResolver(urdf_path)
        self.runtime = ControlRuntime(resolver, normalized_velocity_config=normalized_velocity_config)

        self.get_logger().info(f"Subscribing to Cartesian commands on: {cartesian_cmd_topic}")
        self.get_logger().info(f"Subscribing to Joint States on: {state_feedback_topic}")
        self.get_logger().info(f"Publishing {self.cmd_msg_type} to: {joint_cmd_topic}")
        if raw_action_topic:
            self.get_logger().info(f"Subscribing to raw action chunks on: {raw_action_topic}")

        self.create_subscription(JointState, state_feedback_topic, self.state_cb, 10)
        self.create_subscription(EEFDeltaCommand, cartesian_cmd_topic, self.cmd_cb, 10)
        if raw_action_topic:
            self.create_subscription(RawActionChunk, raw_action_topic, self.raw_action_cb, 10)

        if self.cmd_msg_type == "JointState":
            self.pub_cmd = self.create_publisher(JointState, joint_cmd_topic, 10)
        else:
            self.pub_cmd = self.create_publisher(JointTrajectory, joint_cmd_topic, 10)

        self._dispatch_timer = self.create_timer(0.01, self._dispatch_pending_chunk)

    def state_cb(self, msg: JointState) -> None:
        """Update the latest joint-state cache."""
        for name, pos in zip(msg.name, msg.position, strict=False):
            self.current_joints[name] = pos

    def cmd_cb(self, msg: EEFDeltaCommand) -> None:
        """Resolve a single EEF delta command immediately."""
        if not self.current_joints:
            return

        try:
            command = self.runtime.resolve_eef_delta(
                eef_delta_command_to_array(msg).tolist(),
                self.current_joints,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to resolve EEF delta command: {exc}")
            return

        self._publish_command(command)

    def raw_action_cb(self, msg: RawActionChunk) -> None:
        """Queue a raw action chunk for step-wise dispatch."""
        try:
            chunk = raw_action_chunk_message_to_action_chunk(msg)
            normalized_steps = tuple(
                RawActionStep(
                    step.values,
                    protocol=step.protocol or self._default_raw_action_protocol,
                    frame=step.frame or self._default_raw_action_frame,
                )
                for step in chunk.steps
            )
            normalized_chunk = ActionChunk(
                steps=normalized_steps,
                step_duration_sec=chunk.step_duration_sec,
                metadata=chunk.metadata,
            )
            self.runtime.queue_action_chunk(normalized_chunk)
        except Exception as exc:
            self.get_logger().error(f"Failed to queue raw action chunk: {exc}")

    def _dispatch_pending_chunk(self) -> None:
        if not self.current_joints:
            return

        try:
            command = self.runtime.dispatch_ready(self.current_joints)
        except Exception as exc:
            self.get_logger().error(f"Failed to dispatch raw action chunk step: {exc}")
            return

        if command is not None:
            self._publish_command(command)

    def _publish_command(self, command) -> None:
        if self.cmd_msg_type == "JointState":
            out_msg = JointState()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.name = command.joint_names
            out_msg.position = command.positions
            self.pub_cmd.publish(out_msg)
            return

        out_msg = JointTrajectory()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.joint_names = command.joint_names

        point = JointTrajectoryPoint()
        point.positions = command.positions
        point.time_from_start.sec = 0
        point.time_from_start.nanosec = 500000000

        out_msg.points.append(point)
        self.pub_cmd.publish(out_msg)


class AutoIKNode(ControlRuntimeNode):
    """Backward-compatible alias for the previous control node name."""


def _ros_worker(
    urdf_path: str,
    cartesian_cmd_topic: str,
    state_feedback_topic: str,
    joint_cmd_topic: str,
    cmd_msg_type: str,
    raw_action_topic: str,
    raw_action_protocol: str,
    raw_action_frame: str,
    max_linear_delta: float,
    max_rotation_delta: float,
    invert_gripper_action: bool,
) -> None:
    """Worker function to initialize and run the control host in a separate process."""

    rclpy.init()
    node = ControlRuntimeNode(
        urdf_path=urdf_path,
        cartesian_cmd_topic=cartesian_cmd_topic,
        state_feedback_topic=state_feedback_topic,
        joint_cmd_topic=joint_cmd_topic,
        cmd_msg_type=cmd_msg_type,
        raw_action_topic=raw_action_topic,
        raw_action_protocol=raw_action_protocol,
        raw_action_frame=raw_action_frame,
        max_linear_delta=max_linear_delta,
        max_rotation_delta=max_rotation_delta,
        invert_gripper_action=invert_gripper_action,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


@mcp.tool()
def start_controller(
    urdf_path: str,
    cartesian_cmd_topic: str = EEF_DELTA_CMD_TOPIC,
    state_feedback_topic: str = "/isaac_joint_states",
    joint_cmd_topic: str = "/isaac_joint_commands",
    cmd_msg_type: str = "JointState",
    raw_action_topic: str = RAW_ACTION_CHUNK_TOPIC,
    raw_action_protocol: str = DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    raw_action_frame: str = "tool",
    max_linear_delta: float = 0.075,
    max_rotation_delta: float = 0.15,
    invert_gripper_action: bool = False,
) -> str:
    """
    [ACTION/CONTROL] Start the RoboNeuron control runtime host.

    The runtime consumes canonical EEF delta commands and optionally raw action chunks,
    resolves them against the configured URDF, and publishes joint commands for the robot.
    """
    global _CONTROL_PROCESS
    if _CONTROL_PROCESS is not None and _CONTROL_PROCESS.is_alive():
        return "Error: Controller is already running."

    if cmd_msg_type not in ["JointTrajectory", "JointState"]:
        return "Error: cmd_msg_type must be 'JointTrajectory' or 'JointState'."

    try:
        with open(urdf_path, encoding="utf-8") as handle:
            txt = handle.read()
            if "<robot" not in txt:
                return f"Error: file '{urdf_path}' doesn't look like a URDF (no <robot> tag)."
    except Exception as exc:
        return f"Error: cannot read urdf_path '{urdf_path}': {exc}"

    ctx = multiprocessing.get_context("spawn")
    _CONTROL_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(
            urdf_path,
            cartesian_cmd_topic,
            state_feedback_topic,
            joint_cmd_topic,
            cmd_msg_type,
            raw_action_topic,
            raw_action_protocol,
            raw_action_frame,
            max_linear_delta,
            max_rotation_delta,
            invert_gripper_action,
        ),
        daemon=False,
    )
    _CONTROL_PROCESS.start()
    return (
        f"Success: Controller started with {urdf_path} "
        f"(pid={_CONTROL_PROCESS.pid}, type={cmd_msg_type})."
    )


@mcp.tool()
def stop_controller() -> str:
    """Stop the running control runtime process."""
    global _CONTROL_PROCESS
    if _CONTROL_PROCESS is None or not _CONTROL_PROCESS.is_alive():
        return "Info: No controller is running."

    _CONTROL_PROCESS.terminate()
    _CONTROL_PROCESS.join(timeout=5.0)
    if _CONTROL_PROCESS.is_alive():
        with suppress(Exception):
            _CONTROL_PROCESS.kill()
        _CONTROL_PROCESS.join(timeout=1.0)
    _CONTROL_PROCESS = None
    return "Success: Controller stopped."


if __name__ == "__main__":
    import argparse
    import select
    import sys

    parser = argparse.ArgumentParser(description="control_server.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--urdf", type=str, default="urdf/panda.urdf", help="URDF path to test")
    parser.add_argument("--cartesian-cmd-topic", type=str, default=EEF_DELTA_CMD_TOPIC, help="Topic for EEF delta commands")
    parser.add_argument("--state-feedback-topic", type=str, default="/isaac_joint_states", help="Topic for robot joint state feedback")
    parser.add_argument("--joint-cmd-topic", type=str, default="/isaac_joint_commands", help="Topic for publishing joint commands")
    parser.add_argument("--cmd-msg-type", type=str, default="JointState", choices=["JointTrajectory", "JointState"], help="Output message type")
    parser.add_argument("--raw-action-topic", type=str, default=RAW_ACTION_CHUNK_TOPIC, help="Topic for chunked raw actions")
    parser.add_argument("--raw-action-protocol", type=str, default=DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL, help="Protocol name for raw action chunks")
    parser.add_argument("--raw-action-frame", type=str, default="tool", help="Frame for raw action chunks")
    parser.add_argument("--max-linear-delta", type=float, default=0.075, help="Maximum linear delta for normalized velocity actions")
    parser.add_argument("--max-rotation-delta", type=float, default=0.15, help="Maximum rotational delta for normalized velocity actions")
    parser.add_argument("--invert-gripper-action", action="store_true", help="Invert raw gripper actions before resolving them")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: attempting to start controller (spawn)...")
        res = start_controller(
            args.urdf,
            args.cartesian_cmd_topic,
            args.state_feedback_topic,
            args.joint_cmd_topic,
            args.cmd_msg_type,
            args.raw_action_topic,
            args.raw_action_protocol,
            args.raw_action_frame,
            args.max_linear_delta,
            args.max_rotation_delta,
            args.invert_gripper_action,
        )
        print(res)
        if res.startswith("Error"):
            sys.exit(1)

        try:
            print("Controller started. Press Ctrl-C to stop, or type 'stop' + Enter.")
            while True:
                time.sleep(0.5)
                if _CONTROL_PROCESS is None or not _CONTROL_PROCESS.is_alive():
                    print("Controller process exited.")
                    break
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line.lower() in ("stop", "q", "quit", "exit"):
                        print("Stop command received.")
                        break
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received, stopping controller...")
        finally:
            print(stop_controller())
            print("Local test finished.")
    else:
        mcp.run()
