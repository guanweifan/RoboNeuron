#!/usr/bin/env python3
"""RoboNeuron edge control server."""

from __future__ import annotations

import json
import multiprocessing
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from mcp.server.fastmcp import FastMCP
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from roboneuron_interfaces.msg import EEFDeltaCommand, RawActionChunk, TaskSpaceState
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from roboneuron_backends.franka import backend_metadata_for_robot_profile
from roboneuron_core.kernel import (
    DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL,
    TASK_SPACE_STATE_SOURCE,
    ActionChunk,
    ActionContract,
    ExecutionSession,
    HealthStatus,
    NormalizedCartesianVelocityConfig,
    RawActionStep,
    RuntimeProfile,
)
from roboneuron_core.utils.eef_delta import EEF_DELTA_CMD_TOPIC, eef_delta_command_to_array
from roboneuron_core.utils.raw_action_chunk import (
    RAW_ACTION_CHUNK_TOPIC,
    raw_action_chunk_message_to_action_chunk,
)
from roboneuron_core.utils.task_space_state import array_to_task_space_state_message
from roboneuron_edge.runtime.control_runtime import (
    ControlRuntime,
    URDFKinematicsResolver,
)
from roboneuron_edge.state.task_space_alignment import (
    extract_gripper_open_fraction_from_joint_state,
    pose_matrix_to_state_vector,
    pose_and_gripper_to_state_vector,
)

_CONTROL_PROCESS = None
_CONTROL_SESSION: ExecutionSession | None = None
_CONTROL_HEALTH = HealthStatus.idle("roboneuron-control")
mcp = FastMCP("roboneuron-control")

DEFAULT_CARTESIAN_CMD_TOPIC = EEF_DELTA_CMD_TOPIC
DEFAULT_STATE_FEEDBACK_TOPIC = "/isaac_joint_states"
DEFAULT_JOINT_CMD_TOPIC = "/isaac_joint_commands"
DEFAULT_CMD_MSG_TYPE = "JointState"
DEFAULT_RAW_ACTION_PROTOCOL = DEFAULT_NORMALIZED_CARTESIAN_VELOCITY_PROTOCOL
DEFAULT_RAW_ACTION_FRAME = "tool"
DEFAULT_MAX_LINEAR_DELTA = 0.075
DEFAULT_MAX_ROTATION_DELTA = 0.15
DEFAULT_TRAJECTORY_TIME_FROM_START_SEC = 0.5
DEFAULT_STATE_FEEDBACK_TIMEOUT_SEC = 0.5
DEFAULT_TASK_SPACE_FRAME_ID = "base"
DEFAULT_GRIPPER_COMMAND_MODE = "width"
DEFAULT_GRIPPER_STATE_OPEN_POSITION = 0.04
DEFAULT_GRIPPER_STATE_CLOSED_POSITION = 0.0
DEFAULT_GRIPPER_ACTION_OPEN_POSITION = 0.08
DEFAULT_GRIPPER_ACTION_CLOSED_POSITION = 0.0
DEFAULT_GRIPPER_MAX_EFFORT = 20.0


def _load_gripper_command_action() -> type[Any]:
    try:
        from control_msgs.action import GripperCommand
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "control_msgs is required when gripper_action_name is configured. "
            "Install the ROS 2 Jazzy control message package on this machine."
        ) from exc
    return GripperCommand


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def _resolve_repo_path(path: str | None) -> str | None:
    if path is None:
        return None

    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((_project_root() / candidate).resolve())


def _load_robot_profile(robot_profile: str, config_path: str | None = None) -> dict[str, Any]:
    cfg_path = Path(config_path) if config_path is not None else (_project_root() / "configs" / "robot_profiles.json")
    if not cfg_path.is_absolute():
        cfg_path = (_project_root() / cfg_path).resolve()

    with cfg_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Robot profile config must be a JSON object, got {type(data)}.")
    profile = data.get(robot_profile)
    if not isinstance(profile, dict):
        raise ValueError(f"Robot profile '{robot_profile}' not found in {cfg_path}.")
    return profile
def _resolve_controller_settings(
    *,
    robot_profile: str | None,
    config_path: str | None,
    urdf_path: str | None,
    cartesian_cmd_topic: str | None,
    state_feedback_topic: str | None,
    joint_cmd_topic: str | None,
    cmd_msg_type: str | None,
    raw_action_topic: str | None,
    raw_action_protocol: str | None,
    raw_action_frame: str | None,
    max_linear_delta: float | None,
    max_rotation_delta: float | None,
    invert_gripper_action: bool | None,
    trajectory_time_from_start_sec: float | None,
    state_feedback_timeout_sec: float | None,
    task_space_state_topic: str | None,
    pose_feedback_topic: str | None,
    gripper_state_topic: str | None,
    task_space_frame_id: str | None,
    gripper_action_name: str | None,
    gripper_command_mode: str | None,
    gripper_state_open_position: float | None,
    gripper_state_closed_position: float | None,
    gripper_action_open_position: float | None,
    gripper_action_closed_position: float | None,
    gripper_max_effort: float | None,
    gripper_joint_names: list[str] | None,
) -> dict[str, Any]:
    profile = _load_robot_profile(robot_profile, config_path) if robot_profile is not None else {}
    task_space_profile = profile.get("task_space_state", {}) if isinstance(profile.get("task_space_state"), dict) else {}
    gripper_profile = profile.get("gripper", {}) if isinstance(profile.get("gripper"), dict) else {}

    resolved = {
        "urdf_path": _resolve_repo_path(urdf_path or profile.get("urdf_path")),
        "cartesian_cmd_topic": cartesian_cmd_topic or profile.get("cartesian_cmd_topic") or DEFAULT_CARTESIAN_CMD_TOPIC,
        "state_feedback_topic": state_feedback_topic or profile.get("state_feedback_topic") or DEFAULT_STATE_FEEDBACK_TOPIC,
        "joint_cmd_topic": joint_cmd_topic or profile.get("joint_cmd_topic") or DEFAULT_JOINT_CMD_TOPIC,
        "cmd_msg_type": cmd_msg_type or profile.get("cmd_msg_type") or DEFAULT_CMD_MSG_TYPE,
        "raw_action_topic": raw_action_topic if raw_action_topic is not None else profile.get("raw_action_topic", RAW_ACTION_CHUNK_TOPIC),
        "raw_action_protocol": raw_action_protocol or profile.get("raw_action_protocol") or DEFAULT_RAW_ACTION_PROTOCOL,
        "raw_action_frame": raw_action_frame or profile.get("raw_action_frame") or DEFAULT_RAW_ACTION_FRAME,
        "max_linear_delta": float(
            profile.get("max_linear_delta", DEFAULT_MAX_LINEAR_DELTA)
            if max_linear_delta is None
            else max_linear_delta
        ),
        "max_rotation_delta": float(
            profile.get("max_rotation_delta", DEFAULT_MAX_ROTATION_DELTA)
            if max_rotation_delta is None
            else max_rotation_delta
        ),
        "invert_gripper_action": bool(
            profile.get("invert_gripper_action", False) if invert_gripper_action is None else invert_gripper_action
        ),
        "trajectory_time_from_start_sec": float(
            profile.get("trajectory_time_from_start_sec", DEFAULT_TRAJECTORY_TIME_FROM_START_SEC)
            if trajectory_time_from_start_sec is None
            else trajectory_time_from_start_sec
        ),
        "state_feedback_timeout_sec": float(
            profile.get("state_feedback_timeout_sec", DEFAULT_STATE_FEEDBACK_TIMEOUT_SEC)
            if state_feedback_timeout_sec is None
            else state_feedback_timeout_sec
        ),
        "task_space_state_topic": (
            task_space_state_topic
            if task_space_state_topic is not None
            else task_space_profile.get("topic")
        ),
        "pose_feedback_topic": pose_feedback_topic or task_space_profile.get("pose_feedback_topic"),
        "gripper_state_topic": gripper_state_topic or task_space_profile.get("gripper_state_topic"),
        "task_space_frame_id": task_space_frame_id or task_space_profile.get("frame_id") or DEFAULT_TASK_SPACE_FRAME_ID,
        "gripper_action_name": gripper_action_name or gripper_profile.get("action_name"),
        "gripper_command_mode": gripper_command_mode or gripper_profile.get("command_mode") or DEFAULT_GRIPPER_COMMAND_MODE,
        "gripper_state_open_position": float(
            gripper_profile.get("state_open_position", DEFAULT_GRIPPER_STATE_OPEN_POSITION)
            if gripper_state_open_position is None
            else gripper_state_open_position
        ),
        "gripper_state_closed_position": float(
            gripper_profile.get("state_closed_position", DEFAULT_GRIPPER_STATE_CLOSED_POSITION)
            if gripper_state_closed_position is None
            else gripper_state_closed_position
        ),
        "gripper_action_open_position": float(
            gripper_profile.get("action_open_position", DEFAULT_GRIPPER_ACTION_OPEN_POSITION)
            if gripper_action_open_position is None
            else gripper_action_open_position
        ),
        "gripper_action_closed_position": float(
            gripper_profile.get("action_closed_position", DEFAULT_GRIPPER_ACTION_CLOSED_POSITION)
            if gripper_action_closed_position is None
            else gripper_action_closed_position
        ),
        "gripper_max_effort": float(
            gripper_profile.get("max_effort", DEFAULT_GRIPPER_MAX_EFFORT)
            if gripper_max_effort is None
            else gripper_max_effort
        ),
        "gripper_joint_names": list(gripper_joint_names if gripper_joint_names is not None else gripper_profile.get("joint_names", [])),
    }

    if resolved["urdf_path"] is None:
        raise ValueError("urdf_path is required unless robot_profile provides it.")
    if resolved["cmd_msg_type"] not in {"JointTrajectory", "JointState"}:
        raise ValueError("cmd_msg_type must be 'JointTrajectory' or 'JointState'.")
    if resolved["trajectory_time_from_start_sec"] <= 0:
        raise ValueError("trajectory_time_from_start_sec must be positive.")
    if resolved["state_feedback_timeout_sec"] <= 0:
        raise ValueError("state_feedback_timeout_sec must be positive.")
    if resolved["gripper_command_mode"] not in {"width", "joint_position"}:
        raise ValueError("gripper_command_mode must be 'width' or 'joint_position'.")
    if resolved["task_space_state_topic"] is not None:
        if not resolved["gripper_state_topic"]:
            raise ValueError("gripper_state_topic is required when task_space_state_topic is enabled.")
    return resolved


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
        trajectory_time_from_start_sec: float = DEFAULT_TRAJECTORY_TIME_FROM_START_SEC,
        state_feedback_timeout_sec: float = DEFAULT_STATE_FEEDBACK_TIMEOUT_SEC,
        task_space_state_topic: str | None = None,
        pose_feedback_topic: str | None = None,
        gripper_state_topic: str | None = None,
        task_space_frame_id: str = DEFAULT_TASK_SPACE_FRAME_ID,
        gripper_action_name: str | None = None,
        gripper_command_mode: str = DEFAULT_GRIPPER_COMMAND_MODE,
        gripper_state_open_position: float = DEFAULT_GRIPPER_STATE_OPEN_POSITION,
        gripper_state_closed_position: float = DEFAULT_GRIPPER_STATE_CLOSED_POSITION,
        gripper_action_open_position: float = DEFAULT_GRIPPER_ACTION_OPEN_POSITION,
        gripper_action_closed_position: float = DEFAULT_GRIPPER_ACTION_CLOSED_POSITION,
        gripper_max_effort: float = DEFAULT_GRIPPER_MAX_EFFORT,
        gripper_joint_names: list[str] | None = None,
    ) -> None:
        super().__init__("control_runtime_node")

        self.cmd_msg_type = cmd_msg_type
        self.current_joints: dict[str, float] = {}
        self._default_raw_action_protocol = raw_action_protocol
        self._default_raw_action_frame = raw_action_frame
        self._default_trajectory_time_from_start_sec = float(trajectory_time_from_start_sec)
        self._state_feedback_timeout_sec = float(state_feedback_timeout_sec)
        self._task_space_frame_id = task_space_frame_id
        self._last_joint_state_at: float | None = None
        self._state_feedback_stale_warned = False
        self._task_space_state_pub = (
            self.create_publisher(TaskSpaceState, task_space_state_topic, 10)
            if task_space_state_topic
            else None
        )
        self._use_joint_fk_for_task_space_state = bool(task_space_state_topic and not pose_feedback_topic)
        self._latest_pose_position: list[float] | None = None
        self._latest_pose_orientation: list[float] | None = None
        self._latest_pose_matrix = None
        self._latest_gripper_open_fraction: float | None = None
        self._task_space_pose_ready_logged = False
        self._task_space_gripper_ready_logged = False
        self._task_space_publish_logged = False
        self._raw_chunk_received_logged = False
        self._raw_chunk_dispatch_logged = False
        self._pose_frame_mismatch_warned = False
        self._gripper_joint_names = tuple(gripper_joint_names or [])
        self._gripper_joint_name_set = set(self._gripper_joint_names)
        self._gripper_state_open_position = float(gripper_state_open_position)
        self._gripper_state_closed_position = float(gripper_state_closed_position)
        self._gripper_action_type = _load_gripper_command_action() if gripper_action_name else None
        self._gripper_action_client = (
            ActionClient(self, self._gripper_action_type, gripper_action_name)
            if gripper_action_name
            else None
        )
        self._gripper_command_mode = gripper_command_mode
        self._gripper_action_open_position = float(gripper_action_open_position)
        self._gripper_action_closed_position = float(gripper_action_closed_position)
        self._gripper_max_effort = float(gripper_max_effort)
        self._last_gripper_goal_position: float | None = None
        self._gripper_server_warned = False
        self._gripper_state_from_joint_feedback = (
            gripper_state_topic is not None and gripper_state_topic == state_feedback_topic
        )

        normalized_velocity_config = NormalizedCartesianVelocityConfig(
            max_linear_delta=max_linear_delta,
            max_rotation_delta=max_rotation_delta,
            frame=raw_action_frame,
            invert_gripper=invert_gripper_action,
        )
        resolver = URDFKinematicsResolver(urdf_path)
        self._kinematics_resolver = resolver
        self.runtime = ControlRuntime(resolver, normalized_velocity_config=normalized_velocity_config)

        self.get_logger().info(f"Subscribing to Cartesian commands on: {cartesian_cmd_topic}")
        self.get_logger().info(f"Subscribing to Joint States on: {state_feedback_topic}")
        self.get_logger().info(f"Publishing {self.cmd_msg_type} to: {joint_cmd_topic}")
        if raw_action_topic:
            self.get_logger().info(f"Subscribing to raw action chunks on: {raw_action_topic}")
        if task_space_state_topic:
            self.get_logger().info(f"Publishing task-space state to: {task_space_state_topic}")
        if pose_feedback_topic:
            self.get_logger().info(f"Subscribing to pose feedback on: {pose_feedback_topic}")
        elif self._use_joint_fk_for_task_space_state:
            self.get_logger().info(
                "Deriving task-space pose from joint feedback via local forward kinematics."
            )
        if gripper_state_topic and not self._gripper_state_from_joint_feedback:
            self.get_logger().info(f"Subscribing to gripper state on: {gripper_state_topic}")
        if gripper_action_name:
            self.get_logger().info(f"Sending gripper goals to action: {gripper_action_name}")

        self.create_subscription(JointState, state_feedback_topic, self.state_cb, 10)
        self.create_subscription(EEFDeltaCommand, cartesian_cmd_topic, self.cmd_cb, 10)
        if raw_action_topic:
            self.create_subscription(RawActionChunk, raw_action_topic, self.raw_action_cb, 10)
        if pose_feedback_topic:
            self.create_subscription(PoseStamped, pose_feedback_topic, self.pose_cb, 10)
        if gripper_state_topic and not self._gripper_state_from_joint_feedback:
            self.create_subscription(JointState, gripper_state_topic, self.gripper_state_cb, 10)

        if self.cmd_msg_type == "JointState":
            self.pub_cmd = self.create_publisher(JointState, joint_cmd_topic, 10)
        else:
            self.pub_cmd = self.create_publisher(JointTrajectory, joint_cmd_topic, 10)

        self._dispatch_timer = self.create_timer(0.01, self._dispatch_pending_chunk)

    def state_cb(self, msg: JointState) -> None:
        """Update the latest joint-state cache."""
        for name, pos in zip(msg.name, msg.position, strict=False):
            self.current_joints[name] = pos
        self._last_joint_state_at = time.monotonic()
        self._state_feedback_stale_warned = False
        if self._gripper_state_from_joint_feedback:
            self._update_gripper_open_fraction(msg)
        if self._use_joint_fk_for_task_space_state:
            self._update_pose_from_joint_state()

    def cmd_cb(self, msg: EEFDeltaCommand) -> None:
        """Resolve a single EEF delta command immediately."""
        if not self._has_fresh_joint_state():
            return

        try:
            command = self.runtime.resolve_eef_delta(
                eef_delta_command_to_array(msg).tolist(),
                self.current_joints,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to resolve EEF delta command: {exc}")
            return

        self._publish_command(command, trajectory_time_from_start_sec=self._default_trajectory_time_from_start_sec)

    def raw_action_cb(self, msg: RawActionChunk) -> None:
        """Queue a raw action chunk for step-wise dispatch."""
        if not self._has_fresh_joint_state():
            return
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
            if not self._raw_chunk_received_logged:
                self.get_logger().info(
                    "Queued first raw action chunk: "
                    f"steps={len(normalized_chunk.steps)} "
                    f"step_duration_sec={normalized_chunk.step_duration_sec:.3f} "
                    f"protocol={normalized_chunk.steps[0].protocol} "
                    f"frame={normalized_chunk.steps[0].frame}"
                )
                self._raw_chunk_received_logged = True
        except Exception as exc:
            self.get_logger().error(f"Failed to queue raw action chunk: {exc}")

    def pose_cb(self, msg: PoseStamped) -> None:
        """Cache the latest task-space pose and publish task-space state when ready."""
        if (
            self._task_space_frame_id
            and msg.header.frame_id
            and msg.header.frame_id != self._task_space_frame_id
        ):
            if not self._pose_frame_mismatch_warned:
                self.get_logger().error(
                    "Ignoring pose feedback because frame_id "
                    f"'{msg.header.frame_id}' != expected '{self._task_space_frame_id}'."
                )
                self._pose_frame_mismatch_warned = True
            return

        self._latest_pose_position = [
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        ]
        self._latest_pose_orientation = [
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
        ]
        self._latest_pose_matrix = None
        self._publish_task_space_state_if_ready()

    def gripper_state_cb(self, msg: JointState) -> None:
        """Cache the latest real gripper state and publish task-space state when ready."""
        self._update_gripper_open_fraction(msg)

    def _dispatch_pending_chunk(self) -> None:
        if not self._has_fresh_joint_state():
            if self.runtime.scheduler.pending_count > 0:
                self.runtime.clear_action_chunk()
                self.get_logger().warning("Dropped pending raw action chunk because robot state feedback went stale.")
            return

        try:
            command = self.runtime.dispatch_ready(self.current_joints)
        except Exception as exc:
            self.get_logger().error(f"Failed to dispatch raw action chunk step: {exc}")
            return

        if command is not None:
            if not self._raw_chunk_dispatch_logged:
                self.get_logger().info("Dispatched first raw action chunk step.")
                self._raw_chunk_dispatch_logged = True
            self._publish_command(
                command,
                trajectory_time_from_start_sec=self.runtime.scheduler.step_duration_sec,
            )

    def _update_gripper_open_fraction(self, msg: JointState) -> None:
        try:
            self._latest_gripper_open_fraction = extract_gripper_open_fraction_from_joint_state(
                msg.name,
                msg.position,
                joint_names=self._gripper_joint_names or None,
                closed_position=self._gripper_state_closed_position,
                open_position=self._gripper_state_open_position,
            )
        except Exception:
            return
        if not self._task_space_gripper_ready_logged:
            self.get_logger().info("Received gripper state for task-space state publication.")
            self._task_space_gripper_ready_logged = True
        self._publish_task_space_state_if_ready()

    def _publish_task_space_state_if_ready(self) -> None:
        if self._task_space_state_pub is None:
            return
        if self._latest_gripper_open_fraction is None:
            return

        if self._latest_pose_matrix is not None:
            state = pose_matrix_to_state_vector(
                self._latest_pose_matrix,
                self._latest_gripper_open_fraction,
            )
        else:
            if self._latest_pose_position is None or self._latest_pose_orientation is None:
                return
            state = pose_and_gripper_to_state_vector(
                self._latest_pose_position,
                self._latest_pose_orientation,
                self._latest_gripper_open_fraction,
            )
        out_msg = array_to_task_space_state_message(state)
        self._task_space_state_pub.publish(out_msg)
        if not self._task_space_publish_logged:
            self.get_logger().info("Published first TaskSpaceState message.")
            self._task_space_publish_logged = True

    def _update_pose_from_joint_state(self) -> None:
        if self._task_space_state_pub is None or not self.current_joints:
            return

        try:
            self._latest_pose_matrix = self._kinematics_resolver.current_end_effector_pose(
                self.current_joints
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to derive task-space pose from joint feedback: {exc}")
            return
        if not self._task_space_pose_ready_logged:
            self.get_logger().info("Derived task-space pose from joint feedback.")
            self._task_space_pose_ready_logged = True

        self._publish_task_space_state_if_ready()

    def _map_gripper_open_fraction_to_goal_position(self, open_fraction: float) -> float:
        clipped = float(max(0.0, min(1.0, open_fraction)))
        if self._gripper_command_mode == "joint_position":
            return self._gripper_action_closed_position + (
                self._gripper_action_open_position - self._gripper_action_closed_position
            ) * clipped
        return self._gripper_action_closed_position + (
            self._gripper_action_open_position - self._gripper_action_closed_position
        ) * clipped

    def _send_gripper_goal_if_needed(self, open_fraction: float | None) -> None:
        if self._gripper_action_client is None or open_fraction is None:
            return

        target_position = self._map_gripper_open_fraction_to_goal_position(open_fraction)
        if (
            self._last_gripper_goal_position is not None
            and abs(target_position - self._last_gripper_goal_position) < 1e-4
        ):
            return

        if not self._gripper_action_client.server_is_ready():
            if not self._gripper_action_client.wait_for_server(timeout_sec=0.5):
                if not self._gripper_server_warned:
                    self.get_logger().warning("Gripper action server is not ready; skipping gripper goal.")
                    self._gripper_server_warned = True
                return
            self._gripper_server_warned = False

        if self._gripper_action_type is None:
            return

        goal = self._gripper_action_type.Goal()
        goal.command.position = target_position
        goal.command.max_effort = self._gripper_max_effort
        self._gripper_action_client.send_goal_async(goal)
        self._last_gripper_goal_position = target_position

    def _split_arm_and_gripper_targets(self, command) -> tuple[list[str], list[float]]:
        if self._gripper_action_client is None or not self._gripper_joint_name_set:
            return list(command.joint_names), list(command.positions)

        filtered = [
            (name, position)
            for name, position in zip(command.joint_names, command.positions, strict=False)
            if name not in self._gripper_joint_name_set
        ]
        if not filtered:
            return [], []
        joint_names, positions = zip(*filtered, strict=False)
        return list(joint_names), list(positions)

    def _publish_command(self, command, *, trajectory_time_from_start_sec: float) -> None:
        arm_joint_names, arm_positions = self._split_arm_and_gripper_targets(command)
        self._send_gripper_goal_if_needed(command.gripper_open_fraction)

        if not arm_joint_names:
            return

        if self.cmd_msg_type == "JointState":
            out_msg = JointState()
            out_msg.header.stamp = self.get_clock().now().to_msg()
            out_msg.name = arm_joint_names
            out_msg.position = arm_positions
            self.pub_cmd.publish(out_msg)
            return

        out_msg = JointTrajectory()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.joint_names = arm_joint_names

        point = JointTrajectoryPoint()
        point.positions = arm_positions
        total_nanoseconds = max(1, int(round(float(trajectory_time_from_start_sec) * 1e9)))
        point.time_from_start.sec = total_nanoseconds // 1_000_000_000
        point.time_from_start.nanosec = total_nanoseconds % 1_000_000_000

        out_msg.points.append(point)
        self.pub_cmd.publish(out_msg)

    def _has_fresh_joint_state(self) -> bool:
        if not self.current_joints or self._last_joint_state_at is None:
            if not self._state_feedback_stale_warned:
                self.get_logger().warning("Ignoring control input because no robot joint state feedback has been received yet.")
                self._state_feedback_stale_warned = True
            return False

        if (time.monotonic() - self._last_joint_state_at) <= self._state_feedback_timeout_sec:
            self._state_feedback_stale_warned = False
            return True

        if not self._state_feedback_stale_warned:
            self.get_logger().warning(
                "Ignoring control input because robot joint state feedback is stale "
                f"(>{self._state_feedback_timeout_sec:.3f}s)."
            )
            self._state_feedback_stale_warned = True
        return False


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
    trajectory_time_from_start_sec: float,
    state_feedback_timeout_sec: float,
    task_space_state_topic: str | None,
    pose_feedback_topic: str | None,
    gripper_state_topic: str | None,
    task_space_frame_id: str,
    gripper_action_name: str | None,
    gripper_command_mode: str,
    gripper_state_open_position: float,
    gripper_state_closed_position: float,
    gripper_action_open_position: float,
    gripper_action_closed_position: float,
    gripper_max_effort: float,
    gripper_joint_names: list[str] | None,
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
        trajectory_time_from_start_sec=trajectory_time_from_start_sec,
        state_feedback_timeout_sec=state_feedback_timeout_sec,
        task_space_state_topic=task_space_state_topic,
        pose_feedback_topic=pose_feedback_topic,
        gripper_state_topic=gripper_state_topic,
        task_space_frame_id=task_space_frame_id,
        gripper_action_name=gripper_action_name,
        gripper_command_mode=gripper_command_mode,
        gripper_state_open_position=gripper_state_open_position,
        gripper_state_closed_position=gripper_state_closed_position,
        gripper_action_open_position=gripper_action_open_position,
        gripper_action_closed_position=gripper_action_closed_position,
        gripper_max_effort=gripper_max_effort,
        gripper_joint_names=gripper_joint_names,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        with suppress(Exception):
            node.destroy_node()
        with suppress(Exception):
            rclpy.shutdown()


@mcp.tool()
def start_controller(
    urdf_path: str | None = None,
    cartesian_cmd_topic: str | None = None,
    state_feedback_topic: str | None = None,
    joint_cmd_topic: str | None = None,
    cmd_msg_type: str | None = None,
    raw_action_topic: str | None = None,
    raw_action_protocol: str | None = None,
    raw_action_frame: str | None = None,
    max_linear_delta: float | None = None,
    max_rotation_delta: float | None = None,
    invert_gripper_action: bool | None = None,
    trajectory_time_from_start_sec: float | None = None,
    state_feedback_timeout_sec: float | None = None,
    task_space_state_topic: str | None = None,
    pose_feedback_topic: str | None = None,
    gripper_state_topic: str | None = None,
    task_space_frame_id: str | None = None,
    gripper_action_name: str | None = None,
    gripper_command_mode: str | None = None,
    gripper_state_open_position: float | None = None,
    gripper_state_closed_position: float | None = None,
    gripper_action_open_position: float | None = None,
    gripper_action_closed_position: float | None = None,
    gripper_max_effort: float | None = None,
    gripper_joint_names: list[str] | None = None,
    robot_profile: str | None = None,
    config_path: str | None = None,
) -> str:
    """
    [ACTION/CONTROL] Start the RoboNeuron control runtime host.

    The runtime consumes canonical EEF delta commands and optionally raw action chunks,
    resolves them against the configured URDF, and publishes joint commands for the robot.
    """
    global _CONTROL_HEALTH, _CONTROL_PROCESS, _CONTROL_SESSION
    if _CONTROL_PROCESS is not None and _CONTROL_PROCESS.is_alive():
        return "Error: Controller is already running."

    try:
        resolved = _resolve_controller_settings(
            robot_profile=robot_profile,
            config_path=config_path,
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
            trajectory_time_from_start_sec=trajectory_time_from_start_sec,
            state_feedback_timeout_sec=state_feedback_timeout_sec,
            task_space_state_topic=task_space_state_topic,
            pose_feedback_topic=pose_feedback_topic,
            gripper_state_topic=gripper_state_topic,
            task_space_frame_id=task_space_frame_id,
            gripper_action_name=gripper_action_name,
            gripper_command_mode=gripper_command_mode,
            gripper_state_open_position=gripper_state_open_position,
            gripper_state_closed_position=gripper_state_closed_position,
            gripper_action_open_position=gripper_action_open_position,
            gripper_action_closed_position=gripper_action_closed_position,
            gripper_max_effort=gripper_max_effort,
            gripper_joint_names=gripper_joint_names,
        )
    except Exception as exc:
        _CONTROL_HEALTH = HealthStatus.error("roboneuron-control", summary=str(exc))
        return f"Error: {exc}"

    try:
        with open(resolved["urdf_path"], encoding="utf-8") as handle:
            txt = handle.read()
            if "<robot" not in txt:
                return f"Error: file '{resolved['urdf_path']}' doesn't look like a URDF (no <robot> tag)."
    except Exception as exc:
        _CONTROL_HEALTH = HealthStatus.error(
            "roboneuron-control",
            summary=f"cannot read urdf_path '{resolved['urdf_path']}'",
            details={"error": str(exc)},
        )
        return f"Error: cannot read urdf_path '{resolved['urdf_path']}': {exc}"

    ctx = multiprocessing.get_context("spawn")
    _CONTROL_PROCESS = ctx.Process(
        target=_ros_worker,
        args=(
            resolved["urdf_path"],
            resolved["cartesian_cmd_topic"],
            resolved["state_feedback_topic"],
            resolved["joint_cmd_topic"],
            resolved["cmd_msg_type"],
            resolved["raw_action_topic"],
            resolved["raw_action_protocol"],
            resolved["raw_action_frame"],
            resolved["max_linear_delta"],
            resolved["max_rotation_delta"],
            resolved["invert_gripper_action"],
            resolved["trajectory_time_from_start_sec"],
            resolved["state_feedback_timeout_sec"],
            resolved["task_space_state_topic"],
            resolved["pose_feedback_topic"],
            resolved["gripper_state_topic"],
            resolved["task_space_frame_id"],
            resolved["gripper_action_name"],
            resolved["gripper_command_mode"],
            resolved["gripper_state_open_position"],
            resolved["gripper_state_closed_position"],
            resolved["gripper_action_open_position"],
            resolved["gripper_action_closed_position"],
            resolved["gripper_max_effort"],
            resolved["gripper_joint_names"],
        ),
        daemon=False,
    )
    action_contract = ActionContract.raw_action_chunk(
        protocol=resolved["raw_action_protocol"],
        frame=resolved["raw_action_frame"],
    )
    robot_backend, vendor_stack = backend_metadata_for_robot_profile(robot_profile)
    runtime_profile = RuntimeProfile.edge_control(
        name=robot_profile or "control_runtime",
        deployment_mode="local",
        robot_backend=robot_backend,
        action_transport=action_contract.transport,
        action_protocol=action_contract.protocol,
        state_source=(
            TASK_SPACE_STATE_SOURCE if resolved["task_space_state_topic"] is not None else None
        ),
        vendor_stack=vendor_stack,
        metadata={
            "cmd_msg_type": resolved["cmd_msg_type"],
            "joint_cmd_topic": resolved["joint_cmd_topic"],
        },
    )
    _CONTROL_SESSION = ExecutionSession.create(
        owner="roboneuron-control",
        action_contract=action_contract,
        runtime_profile=runtime_profile,
        metadata={
            "cmd_msg_type": resolved["cmd_msg_type"],
            "joint_cmd_topic": resolved["joint_cmd_topic"],
            "robot_profile": robot_profile,
        },
    )
    _CONTROL_SESSION.mark_starting(
        details={
            "raw_action_topic": resolved["raw_action_topic"],
            "task_space_state_topic": resolved["task_space_state_topic"],
        }
    )
    _CONTROL_PROCESS.start()
    _CONTROL_SESSION.mark_running(
        details={
            "pid": _CONTROL_PROCESS.pid,
            "robot_profile": robot_profile,
        }
    )
    _CONTROL_HEALTH = HealthStatus.ready(
        "roboneuron-control",
        summary="control runtime is running",
        details={
            "pid": _CONTROL_PROCESS.pid,
            "cmd_msg_type": resolved["cmd_msg_type"],
            "robot_profile": robot_profile,
            "profile_name": runtime_profile.name,
            "runtime_layer": runtime_profile.layer,
        },
    )
    return (
        f"Success: Controller started with {resolved['urdf_path']} "
        f"(pid={_CONTROL_PROCESS.pid}, type={resolved['cmd_msg_type']})."
    )


@mcp.tool()
def stop_controller() -> str:
    """Stop the running control runtime process."""
    global _CONTROL_HEALTH, _CONTROL_PROCESS, _CONTROL_SESSION
    if _CONTROL_PROCESS is None or not _CONTROL_PROCESS.is_alive():
        return "Info: No controller is running."

    _CONTROL_PROCESS.terminate()
    _CONTROL_PROCESS.join(timeout=5.0)
    if _CONTROL_PROCESS.is_alive():
        with suppress(Exception):
            _CONTROL_PROCESS.kill()
        _CONTROL_PROCESS.join(timeout=1.0)
    _CONTROL_PROCESS = None
    if _CONTROL_SESSION is not None:
        _CONTROL_SESSION.mark_stopped(
            reason="stop_controller requested",
            details={"requested_by": "mcp"},
        )
    _CONTROL_HEALTH = HealthStatus.idle("roboneuron-control")
    return "Success: Controller stopped."


if __name__ == "__main__":
    import argparse
    import select
    import sys

    parser = argparse.ArgumentParser(description="control_server.py local test harness")
    parser.add_argument("--local-test", action="store_true", help="Run local start/stop test instead of MCP server")
    parser.add_argument("--robot-profile", type=str, default=None, help="Robot profile name from configs/robot_profiles.json")
    parser.add_argument("--config-path", type=str, default=None, help="Optional robot profile config path")
    parser.add_argument("--urdf", type=str, default=None, help="URDF path to test")
    parser.add_argument("--cartesian-cmd-topic", type=str, default=None, help="Topic for EEF delta commands")
    parser.add_argument("--state-feedback-topic", type=str, default=None, help="Topic for robot joint state feedback")
    parser.add_argument("--joint-cmd-topic", type=str, default=None, help="Topic for publishing joint commands")
    parser.add_argument("--cmd-msg-type", type=str, default=None, choices=["JointTrajectory", "JointState"], help="Output message type")
    parser.add_argument("--raw-action-topic", type=str, default=None, help="Topic for chunked raw actions")
    parser.add_argument("--raw-action-protocol", type=str, default=None, help="Protocol name for raw action chunks")
    parser.add_argument("--raw-action-frame", type=str, default=None, help="Frame for raw action chunks")
    parser.add_argument("--max-linear-delta", type=float, default=None, help="Maximum linear delta for normalized velocity actions")
    parser.add_argument("--max-rotation-delta", type=float, default=None, help="Maximum rotational delta for normalized velocity actions")
    parser.add_argument("--invert-gripper-action", action="store_true", help="Invert raw gripper actions before resolving them")
    parser.add_argument("--trajectory-time-from-start-sec", type=float, default=None, help="JointTrajectory point time_from_start in seconds")
    parser.add_argument("--state-feedback-timeout-sec", type=float, default=None, help="Maximum allowed age for robot joint state feedback")
    parser.add_argument("--task-space-state-topic", type=str, default=None, help="Optional TaskSpaceState publish topic")
    parser.add_argument("--pose-feedback-topic", type=str, default=None, help="PoseStamped topic used to build TaskSpaceState")
    parser.add_argument("--gripper-state-topic", type=str, default=None, help="JointState topic used to build TaskSpaceState")
    parser.add_argument("--task-space-frame-id", type=str, default=None, help="Expected frame_id for pose feedback")
    parser.add_argument("--gripper-action-name", type=str, default=None, help="Optional gripper action name, e.g. /franka_gripper/gripper_action")
    parser.add_argument("--gripper-command-mode", type=str, default=None, choices=["width", "joint_position"], help="Gripper action command semantics")
    parser.add_argument("--gripper-state-open-position", type=float, default=None, help="Per-finger open position used to normalize gripper state")
    parser.add_argument("--gripper-state-closed-position", type=float, default=None, help="Per-finger closed position used to normalize gripper state")
    parser.add_argument("--gripper-action-open-position", type=float, default=None, help="Open command position sent to gripper action server")
    parser.add_argument("--gripper-action-closed-position", type=float, default=None, help="Closed command position sent to gripper action server")
    parser.add_argument("--gripper-max-effort", type=float, default=None, help="max_effort sent with gripper action goals")
    parser.add_argument("--gripper-joint-names", nargs="*", default=None, help="Explicit finger joint names for state extraction/filtering")
    args = parser.parse_args()

    if args.local_test:
        print("LOCAL TEST MODE: attempting to start controller (spawn)...")
        res = start_controller(
            urdf_path=args.urdf,
            cartesian_cmd_topic=args.cartesian_cmd_topic,
            state_feedback_topic=args.state_feedback_topic,
            joint_cmd_topic=args.joint_cmd_topic,
            cmd_msg_type=args.cmd_msg_type,
            raw_action_topic=args.raw_action_topic,
            raw_action_protocol=args.raw_action_protocol,
            raw_action_frame=args.raw_action_frame,
            max_linear_delta=args.max_linear_delta,
            max_rotation_delta=args.max_rotation_delta,
            invert_gripper_action=args.invert_gripper_action if args.invert_gripper_action else None,
            trajectory_time_from_start_sec=args.trajectory_time_from_start_sec,
            state_feedback_timeout_sec=args.state_feedback_timeout_sec,
            task_space_state_topic=args.task_space_state_topic,
            pose_feedback_topic=args.pose_feedback_topic,
            gripper_state_topic=args.gripper_state_topic,
            task_space_frame_id=args.task_space_frame_id,
            gripper_action_name=args.gripper_action_name,
            gripper_command_mode=args.gripper_command_mode,
            gripper_state_open_position=args.gripper_state_open_position,
            gripper_state_closed_position=args.gripper_state_closed_position,
            gripper_action_open_position=args.gripper_action_open_position,
            gripper_action_closed_position=args.gripper_action_closed_position,
            gripper_max_effort=args.gripper_max_effort,
            gripper_joint_names=args.gripper_joint_names,
            robot_profile=args.robot_profile,
            config_path=args.config_path,
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
