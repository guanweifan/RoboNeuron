"""Live ROS test for the EEF delta -> control -> Isaac Sim command chain."""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_ROS_LOG_DIR = Path("/tmp/roboneuron_ros_logs")
DEFAULT_ROS_LOG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("ROS_LOG_DIR", str(DEFAULT_ROS_LOG_DIR))

rclpy = pytest.importorskip(
    "rclpy",
    reason="ROS 2 runtime not available. Source /opt/ros/jazzy/setup.bash first.",
)
pytest.importorskip(
    "roboneuron_interfaces.msg",
    reason="roboneuron_interfaces is not available. Source the ROS workspace that provides EEFDeltaCommand first.",
)
pytest.importorskip(
    "sensor_msgs.msg",
    reason="sensor_msgs is not available. Source /opt/ros/jazzy/setup.bash first.",
)

from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState

from roboneuron_core.utils.eef_delta import EEF_DELTA_CMD_TOPIC

pytestmark = [pytest.mark.integration, pytest.mark.ros, pytest.mark.e2e]


def _wait_until(predicate: Any, executor: SingleThreadedExecutor, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
        if predicate():
            return True
    return False


def _ensure_ros_node(module: ModuleType, skip_message: str) -> None:
    if getattr(module, "ros_node", None) is not None:
        return

    init_ros_node = getattr(module, "init_ros_node", None)
    if callable(init_ros_node):
        if rclpy.ok():
            rclpy.shutdown()
        init_ros_node()

    if getattr(module, "ros_node", None) is None:
        pytest.skip(skip_message)


@pytest.fixture
def eef_delta_server_module() -> Any:
    module_name = "roboneuron_core.servers.generated.eef_delta_server"
    existing = sys.modules.get(module_name)
    if existing is not None:
        old_node = getattr(existing, "ros_node", None)
        if old_node is not None:
            old_node.destroy_node()
            existing.ros_node = None
        if rclpy.ok():
            rclpy.shutdown()
        del sys.modules[module_name]

    module: ModuleType = importlib.import_module(module_name)
    _ensure_ros_node(module, "eef_delta_server could not initialize its ROS node in this environment.")
    yield module

    ros_node = getattr(module, "ros_node", None)
    if ros_node is not None:
        ros_node.destroy_node()
        module.ros_node = None
    if rclpy.ok():
        rclpy.shutdown()


def test_eef_delta_drives_live_isaac_joint_command_output(
    eef_delta_server_module: Any,
) -> None:
    if not rclpy.ok():
        rclpy.init()

    from roboneuron_core.servers.control_server import AutoIKNode

    urdf_path = str(ROOT / "urdf" / "panda.urdf")
    control_node = AutoIKNode(
        urdf_path=urdf_path,
        cartesian_cmd_topic=EEF_DELTA_CMD_TOPIC,
        state_feedback_topic="/isaac_joint_states",
        joint_cmd_topic="/isaac_joint_commands",
        cmd_msg_type="JointState",
    )
    command_listener = rclpy.create_node(f"test_live_isaac_joint_cmd_listener_{time.time_ns()}")

    received: list[JointState] = []
    subscription = command_listener.create_subscription(
        JointState,
        "/isaac_joint_commands",
        lambda msg: received.append(msg),
        10,
    )

    executor = SingleThreadedExecutor()
    for node in [eef_delta_server_module.ros_node, control_node, command_listener]:
        executor.add_node(node)

    try:
        if not _wait_until(lambda: bool(control_node.current_joints), executor, timeout_sec=8.0):
            pytest.skip(
                "No live JointState feedback received on /isaac_joint_states. "
                "Start Isaac Sim with the ROS joint state bridge enabled first."
            )

        payload = eef_delta_server_module.EEFDeltaInput(
            delta_x=0.01,
            delta_y=0.0,
            delta_z=0.0,
            delta_roll=0.0,
            delta_pitch=0.0,
            delta_yaw=0.0,
            gripper_cmd=0.5,
        )
        result = eef_delta_server_module.pub_eef_delta(payload)
        assert result == f"Published to {EEF_DELTA_CMD_TOPIC}"

        assert _wait_until(lambda: bool(received), executor, timeout_sec=5.0), (
            "Did not receive a JointState command on /isaac_joint_commands within 5 seconds."
        )

        message = received[0]
        assert message.name
        assert len(message.position) == len(message.name)
        finger_positions = {
            name: message.position[idx]
            for idx, name in enumerate(message.name)
            if "finger" in name
        }
        if finger_positions:
            assert finger_positions["panda_finger_joint1"] == pytest.approx(0.02)
            assert finger_positions["panda_finger_joint2"] == pytest.approx(0.02)
    finally:
        del subscription
        executor.shutdown()
        command_listener.destroy_node()
        control_node.destroy_node()
