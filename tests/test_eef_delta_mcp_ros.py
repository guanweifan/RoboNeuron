"""ROS integration test for the EEF delta MCP publisher."""

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
    reason="ROS 2 runtime not available. Source /opt/ros/humble/setup.bash first.",
)
pytest.importorskip(
    "roboneuron_interfaces.msg",
    reason="roboneuron_interfaces is not available. Source the ROS workspace that provides EEFDeltaCommand first.",
)

from roboneuron_interfaces.msg import EEFDeltaCommand
from rclpy.executors import SingleThreadedExecutor

pytestmark = [pytest.mark.integration, pytest.mark.ros]
TEST_EEF_DELTA_TOPIC = "/test_eef_delta_cmd"


def _wait_for_messages(
    executor: SingleThreadedExecutor,
    received: list[EEFDeltaCommand],
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
        if received:
            return True
    return False


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

    try:
        module: ModuleType = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(
            f"Failed to import EEF delta MCP server: {exc}. Run this test with "
            "`source /opt/ros/humble/setup.bash && uv run pytest ...`.",
            pytrace=False,
        )

    try:
        if module.ros_node is None:
            pytest.skip(
                "eef_delta_server could not initialize its ROS node. "
                "Ensure DDS transport permissions are available in this environment."
            )
        yield module
    finally:
        ros_node = getattr(module, "ros_node", None)
        if ros_node is not None:
            ros_node.destroy_node()
            module.ros_node = None
        if rclpy.ok():
            rclpy.shutdown()


def test_pub_eef_delta_publishes_to_eef_delta_cmd(eef_delta_server_module: Any) -> None:
    received: list[EEFDeltaCommand] = []
    listener = rclpy.create_node(f"test_eef_delta_listener_{time.time_ns()}")
    subscription = listener.create_subscription(
        EEFDeltaCommand,
        TEST_EEF_DELTA_TOPIC,
        lambda msg: received.append(msg),
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(listener)

    try:
        payload = eef_delta_server_module.EEFDeltaInput(
            delta_x=0.01,
            delta_y=-0.02,
            delta_z=0.03,
            delta_roll=0.1,
            delta_pitch=-0.2,
            delta_yaw=0.3,
            gripper_cmd=1.0,
        )

        # Give DDS discovery a brief window before publishing.
        for _ in range(5):
            executor.spin_once(timeout_sec=0.1)

        result = eef_delta_server_module.pub_eef_delta(payload)
        assert result == f"Published to {TEST_EEF_DELTA_TOPIC}"

        assert _wait_for_messages(executor, received, timeout_sec=3.0), (
            f"Did not receive an EEFDeltaCommand message on {TEST_EEF_DELTA_TOPIC} within 3 seconds."
        )

        msg = received[0]
        print(
            f"received {TEST_EEF_DELTA_TOPIC}:",
            {
                "delta_x": msg.delta_x,
                "delta_y": msg.delta_y,
                "delta_z": msg.delta_z,
                "delta_roll": msg.delta_roll,
                "delta_pitch": msg.delta_pitch,
                "delta_yaw": msg.delta_yaw,
                "gripper_cmd": msg.gripper_cmd,
            },
        )
        assert msg.delta_x == pytest.approx(0.01)
        assert msg.delta_y == pytest.approx(-0.02)
        assert msg.delta_z == pytest.approx(0.03)
        assert msg.delta_roll == pytest.approx(0.1)
        assert msg.delta_pitch == pytest.approx(-0.2)
        assert msg.delta_yaw == pytest.approx(0.3)
        assert msg.gripper_cmd == pytest.approx(1.0)
    finally:
        del subscription
        executor.shutdown()
        listener.destroy_node()
