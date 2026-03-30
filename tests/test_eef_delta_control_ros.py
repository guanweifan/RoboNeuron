"""ROS integration test for the isolated EEF delta -> control IK chain."""

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

pytestmark = [pytest.mark.integration, pytest.mark.ros]
_TEST_TOPIC_SUFFIX = f"{os.getpid()}_{time.time_ns()}"
TEST_EEF_DELTA_TOPIC = f"/test_eef_delta_cmd_{_TEST_TOPIC_SUFFIX}"
TEST_STATE_TOPIC = f"/test_eef_delta_joint_states_{_TEST_TOPIC_SUFFIX}"
TEST_COMMAND_TOPIC = f"/test_eef_delta_joint_commands_{_TEST_TOPIC_SUFFIX}"

PANDA_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
]


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
def eef_delta_server_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    import roboneuron_core.utils.eef_delta as eef_delta_utils

    monkeypatch.setattr(eef_delta_utils, "EEF_DELTA_CMD_TOPIC", TEST_EEF_DELTA_TOPIC)
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


def test_eef_delta_drives_control_server_to_jointstate_output(
    eef_delta_server_module: Any,
) -> None:
    if not rclpy.ok():
        rclpy.init()

    from roboneuron_edge.servers.control_server import AutoIKNode

    urdf_path = str(ROOT / "urdf" / "panda.urdf")
    control_node = AutoIKNode(
        urdf_path=urdf_path,
        cartesian_cmd_topic=TEST_EEF_DELTA_TOPIC,
        state_feedback_topic=TEST_STATE_TOPIC,
        joint_cmd_topic=TEST_COMMAND_TOPIC,
        cmd_msg_type="JointState",
    )
    feedback_pub_node = rclpy.create_node(f"test_joint_state_pub_{time.time_ns()}")
    command_listener = rclpy.create_node(f"test_joint_cmd_listener_{time.time_ns()}")

    feedback_pub = feedback_pub_node.create_publisher(JointState, TEST_STATE_TOPIC, 10)
    received: list[JointState] = []
    subscription = command_listener.create_subscription(
        JointState,
        TEST_COMMAND_TOPIC,
        lambda msg: received.append(msg),
        10,
    )

    executor = SingleThreadedExecutor()
    for node in [eef_delta_server_module.ros_node, control_node, feedback_pub_node, command_listener]:
        executor.add_node(node)

    try:
        initial_state = JointState()
        initial_state.name = PANDA_JOINT_NAMES
        initial_state.position = [0.0] * len(PANDA_JOINT_NAMES)

        for _ in range(5):
            feedback_pub.publish(initial_state)
            executor.spin_once(timeout_sec=0.1)

        assert _wait_until(lambda: bool(control_node.current_joints), executor, timeout_sec=2.0), (
            f"Control node did not receive any JointState feedback on {TEST_STATE_TOPIC}."
        )

        payload = eef_delta_server_module.EEFDeltaInput(
            delta_x=0.02,
            delta_y=0.0,
            delta_z=0.01,
            delta_roll=0.0,
            delta_pitch=0.0,
            delta_yaw=0.0,
            gripper_cmd=0.5,
        )
        result = eef_delta_server_module.pub_eef_delta(payload)
        assert result == f"Published to {TEST_EEF_DELTA_TOPIC}"

        assert _wait_until(lambda: bool(received), executor, timeout_sec=3.0), (
            f"Did not receive a JointState command on {TEST_COMMAND_TOPIC} within 3 seconds."
        )

        message = received[0]
        assert "panda_finger_joint1" in message.name
        assert "panda_finger_joint2" in message.name
        finger_positions = {
            name: message.position[idx]
            for idx, name in enumerate(message.name)
            if "finger" in name
        }
        assert finger_positions["panda_finger_joint1"] == pytest.approx(0.02)
        assert finger_positions["panda_finger_joint2"] == pytest.approx(0.02)
        assert len(message.position) == len(message.name)
    finally:
        del subscription
        executor.shutdown()
        command_listener.destroy_node()
        feedback_pub_node.destroy_node()
        control_node.destroy_node()
