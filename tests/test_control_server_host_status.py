from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from roboneuron_core.kernel import ExecutionSessionStatus, HealthLevel


def _install_fake_control_server_modules(monkeypatch) -> None:
    fake_geometry_msgs = types.ModuleType("geometry_msgs")
    fake_geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    fake_geometry_msgs_msg.PoseStamped = object
    fake_geometry_msgs.msg = fake_geometry_msgs_msg

    fake_mcp = types.ModuleType("mcp")
    fake_mcp_server = types.ModuleType("mcp.server")
    fake_mcp_fastmcp = types.ModuleType("mcp.server.fastmcp")

    class FakeFastMCP:
        def __init__(self, name: str) -> None:
            self.name = name

        def tool(self):
            def _decorator(func):
                return func

            return _decorator

    fake_mcp_fastmcp.FastMCP = FakeFastMCP
    fake_mcp.server = fake_mcp_server
    fake_mcp_server.fastmcp = fake_mcp_fastmcp

    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda *args, **kwargs: None
    fake_rclpy.shutdown = lambda *args, **kwargs: None
    fake_rclpy.spin = lambda *args, **kwargs: None

    fake_rclpy_action = types.ModuleType("rclpy.action")
    fake_rclpy_action.ActionClient = object

    fake_rclpy_executors = types.ModuleType("rclpy.executors")
    fake_rclpy_executors.ExternalShutdownException = RuntimeError

    fake_rclpy_node = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    fake_rclpy_node.Node = FakeNode

    fake_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")
    fake_interfaces_msg.EEFDeltaCommand = object
    fake_interfaces_msg.RawActionChunk = object
    fake_interfaces_msg.TaskSpaceState = object
    fake_interfaces.msg = fake_interfaces_msg

    fake_sensor_msgs = types.ModuleType("sensor_msgs")
    fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    fake_sensor_msgs_msg.JointState = object
    fake_sensor_msgs.msg = fake_sensor_msgs_msg

    fake_trajectory_msgs = types.ModuleType("trajectory_msgs")
    fake_trajectory_msgs_msg = types.ModuleType("trajectory_msgs.msg")
    fake_trajectory_msgs_msg.JointTrajectory = object
    fake_trajectory_msgs_msg.JointTrajectoryPoint = object
    fake_trajectory_msgs.msg = fake_trajectory_msgs_msg

    monkeypatch.setitem(sys.modules, "geometry_msgs", fake_geometry_msgs)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", fake_geometry_msgs_msg)
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)
    monkeypatch.setitem(sys.modules, "mcp.server", fake_mcp_server)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fake_mcp_fastmcp)
    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.action", fake_rclpy_action)
    monkeypatch.setitem(sys.modules, "rclpy.executors", fake_rclpy_executors)
    monkeypatch.setitem(sys.modules, "rclpy.node", fake_rclpy_node)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_interfaces_msg)
    monkeypatch.setitem(sys.modules, "sensor_msgs", fake_sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", fake_sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "trajectory_msgs", fake_trajectory_msgs)
    monkeypatch.setitem(sys.modules, "trajectory_msgs.msg", fake_trajectory_msgs_msg)


def test_start_controller_tracks_session_and_health(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    class FakeProcess:
        def __init__(self, target=None, args=(), daemon=False) -> None:
            del target, args, daemon
            self.pid = 9876
            self._alive = False

        def start(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def kill(self) -> None:
            self._alive = False

    class FakeContext:
        def Process(self, target=None, args=(), daemon=False) -> FakeProcess:  # noqa: N802
            return FakeProcess(target=target, args=args, daemon=daemon)

    monkeypatch.setattr(control_server.multiprocessing, "get_context", lambda method: FakeContext())

    project_root = Path(__file__).resolve().parents[1]
    result = control_server.start_controller(
        urdf_path=str(project_root / "urdf" / "panda.urdf"),
        cartesian_cmd_topic="/eef_delta_cmd",
        state_feedback_topic="/joint_states",
        joint_cmd_topic="/joint_commands",
        cmd_msg_type="JointState",
    )

    assert result == (
        f"Success: Controller started with {project_root / 'urdf' / 'panda.urdf'} "
        "(pid=9876, type=JointState)."
    )
    assert control_server._CONTROL_SESSION is not None
    assert control_server._CONTROL_SESSION.status == ExecutionSessionStatus.RUNNING
    assert control_server._CONTROL_SESSION.runtime_profile is not None
    assert control_server._CONTROL_SESSION.runtime_profile.layer == "edge"
    assert control_server._CONTROL_SESSION.trace is not None
    assert control_server._CONTROL_SESSION.trace.profile_name == "control_runtime"
    assert control_server._CONTROL_HEALTH.level == HealthLevel.READY

    stop_result = control_server.stop_controller()

    assert stop_result == "Success: Controller stopped."
    assert control_server._CONTROL_SESSION.status == ExecutionSessionStatus.STOPPED
    assert control_server._CONTROL_SESSION.history[-1].details["requested_by"] == "mcp"
    assert control_server._CONTROL_HEALTH.level == HealthLevel.IDLE


def test_resolve_controller_settings_allows_task_space_state_without_pose_topic(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    project_root = Path(__file__).resolve().parents[1]
    resolved = control_server._resolve_controller_settings(
        robot_profile=None,
        config_path=None,
        urdf_path=str(project_root / "urdf" / "fr3.urdf"),
        cartesian_cmd_topic="/eef_delta_cmd",
        state_feedback_topic="/franka/joint_states",
        joint_cmd_topic="/fr3_arm_controller/joint_trajectory",
        cmd_msg_type="JointTrajectory",
        raw_action_topic="/raw_action_chunk",
        raw_action_protocol="normalized_cartesian_velocity",
        raw_action_frame="tool",
        max_linear_delta=None,
        max_rotation_delta=None,
        invert_gripper_action=None,
        trajectory_time_from_start_sec=None,
        state_feedback_timeout_sec=None,
        task_space_state_topic="/task_space_state",
        pose_feedback_topic=None,
        gripper_state_topic="/franka_gripper/joint_states",
        task_space_frame_id="base",
        gripper_action_name="/franka_gripper/gripper_action",
        gripper_command_mode="joint_position",
        gripper_state_open_position=None,
        gripper_state_closed_position=None,
        gripper_action_open_position=None,
        gripper_action_closed_position=None,
        gripper_max_effort=None,
        gripper_joint_names=["fr3_finger_joint1", "fr3_finger_joint2"],
    )

    assert resolved["task_space_state_topic"] == "/task_space_state"
    assert resolved["pose_feedback_topic"] is None
    assert resolved["gripper_state_topic"] == "/franka_gripper/joint_states"
