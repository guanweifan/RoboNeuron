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
    fake_geometry_msgs_msg.TwistStamped = object
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

    fake_std_msgs = types.ModuleType("std_msgs")
    fake_std_msgs_msg = types.ModuleType("std_msgs.msg")
    fake_std_msgs_msg.Float64MultiArray = object
    fake_std_msgs.msg = fake_std_msgs_msg

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
    monkeypatch.setitem(sys.modules, "std_msgs", fake_std_msgs)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", fake_std_msgs_msg)
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
        raw_action_dispatch_period_sec=None,
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
    assert resolved["raw_action_dispatch_period_sec"] == control_server.DEFAULT_RAW_ACTION_DISPATCH_PERIOD_SEC


def test_resolve_controller_settings_accepts_twist_stamped_transport(monkeypatch) -> None:
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
        state_feedback_topic="/joint_states",
        joint_cmd_topic="/fr3_arm_controller/cmd_vel",
        cmd_msg_type="TwistStamped",
        raw_action_topic="/raw_action_chunk",
        raw_action_protocol="normalized_cartesian_velocity",
        raw_action_frame="tool",
        max_linear_delta=None,
        max_rotation_delta=None,
        invert_gripper_action=None,
        trajectory_time_from_start_sec=None,
        raw_action_dispatch_period_sec=0.02,
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

    assert resolved["cmd_msg_type"] == "TwistStamped"
    assert resolved["joint_cmd_topic"] == "/fr3_arm_controller/cmd_vel"


def test_joint_position_gripper_goal_keeps_open_side_margin(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    node = object.__new__(control_server.ControlRuntimeNode)
    node._gripper_command_mode = "joint_position"
    node._gripper_action_open_position = 0.039774
    node._gripper_action_closed_position = 0.0

    target = node._map_gripper_open_fraction_to_goal_position(1.0)

    assert 0.0397 < target < 0.039774


def test_joint_position_gripper_goal_uses_binary_thresholds(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    node = object.__new__(control_server.ControlRuntimeNode)
    node._gripper_command_mode = "joint_position"
    node._gripper_action_open_position = 0.039774
    node._gripper_action_closed_position = 0.0
    node._gripper_binary_threshold = control_server.DEFAULT_GRIPPER_BINARY_THRESHOLD
    node._latest_gripper_open_fraction = 0.0
    node._last_gripper_goal_position = None

    assert node._select_gripper_goal_open_fraction(0.5) is None
    assert node._select_gripper_goal_open_fraction(0.95) == 1.0

    node._latest_gripper_open_fraction = 1.0
    assert node._select_gripper_goal_open_fraction(0.95) is None
    assert node._select_gripper_goal_open_fraction(0.05) == 0.0


def test_velocity_blend_state_smoothly_interpolates_targets(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    blend = control_server.VelocityBlendState()
    blend.set_target([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], now=0.0, duration_sec=1.0)

    assert blend.value(now=0.0)[0] == 0.0
    assert blend.value(now=0.5)[0] == 0.5
    assert blend.value(now=1.0)[0] == 1.0
    assert not blend.target_is_zero()

    blend.set_target([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], now=1.0, duration_sec=0.5)
    assert blend.value(now=1.5)[0] == 0.0
    assert blend.target_is_zero()


def test_resolve_controller_settings_uses_fr3_real_profile_defaults(monkeypatch) -> None:
    _install_fake_control_server_modules(monkeypatch)
    module_name = "roboneuron_edge.servers.control_server"
    sys.modules.pop(module_name, None)
    control_server = importlib.import_module(module_name)

    resolved = control_server._resolve_controller_settings(
        robot_profile="fr3_real",
        config_path=None,
        urdf_path=None,
        cartesian_cmd_topic=None,
        state_feedback_topic=None,
        joint_cmd_topic=None,
        cmd_msg_type=None,
        raw_action_topic=None,
        raw_action_protocol=None,
        raw_action_frame=None,
        max_linear_delta=None,
        max_rotation_delta=None,
        invert_gripper_action=None,
        trajectory_time_from_start_sec=None,
        raw_action_dispatch_period_sec=None,
        state_feedback_timeout_sec=None,
        task_space_state_topic=None,
        pose_feedback_topic=None,
        gripper_state_topic=None,
        task_space_frame_id=None,
        gripper_action_name=None,
        gripper_command_mode=None,
        gripper_state_open_position=None,
        gripper_state_closed_position=None,
        gripper_action_open_position=None,
        gripper_action_closed_position=None,
        gripper_max_effort=None,
        gripper_joint_names=None,
    )

    assert resolved["cmd_msg_type"] == "JointTrajectory"
    assert resolved["joint_cmd_topic"] == "/fr3_arm_controller/joint_trajectory"
    assert resolved["raw_action_dispatch_period_sec"] == 0.02
    assert resolved["max_linear_delta"] == 0.075
    assert resolved["max_rotation_delta"] == 0.15
    assert resolved["invert_gripper_action"] is True
    assert resolved["gripper_command_mode"] == "joint_position"
    assert resolved["trajectory_time_from_start_sec"] == 0.02
