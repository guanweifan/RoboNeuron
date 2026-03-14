"""ROS integration test for the Twist MCP publisher."""

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
    "geometry_msgs.msg",
    reason="geometry_msgs is not available. Source /opt/ros/humble/setup.bash first.",
)

from geometry_msgs.msg import Twist
from rclpy.executors import SingleThreadedExecutor

pytestmark = [pytest.mark.integration, pytest.mark.ros]


def _wait_for_messages(
    executor: SingleThreadedExecutor,
    received: list[Twist],
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.1)
        if received:
            return True
    return False


@pytest.fixture
def twist_server_module() -> Any:
    module_name = "roboneuron_core.servers.generated.twist_server"
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
            f"Failed to import Twist MCP server: {exc}. Run this test with "
            "`source /opt/ros/humble/setup.bash && uv run pytest ...`.",
            pytrace=False,
        )

    try:
        if module.ros_node is None:
            pytest.skip(
                "twist_server could not initialize its ROS node. "
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


def test_pub_twist_publishes_to_cmd_vel(twist_server_module: Any) -> None:
    received: list[Twist] = []
    listener = rclpy.create_node(f"test_twist_listener_{time.time_ns()}")
    subscription = listener.create_subscription(
        Twist,
        "/cmd_vel",
        lambda msg: received.append(msg),
        10,
    )
    executor = SingleThreadedExecutor()
    executor.add_node(listener)

    try:
        payload = twist_server_module.TwistInput(
            linear=twist_server_module.Linear(x=0.42, y=0.0, z=0.0),
            angular=twist_server_module.Angular(x=0.0, y=0.0, z=-0.25),
        )

        # Give DDS discovery a brief window before publishing.
        for _ in range(5):
            executor.spin_once(timeout_sec=0.1)

        result = twist_server_module.pub_twist(payload)
        assert result == "Published to /cmd_vel"

        assert _wait_for_messages(executor, received, timeout_sec=3.0), (
            "Did not receive a Twist message on /cmd_vel within 3 seconds."
        )

        msg = received[0]
        print(
            "received /cmd_vel:",
            {
                "linear": (msg.linear.x, msg.linear.y, msg.linear.z),
                "angular": (msg.angular.x, msg.angular.y, msg.angular.z),
            },
        )
        assert msg.linear.x == pytest.approx(0.42)
        assert msg.linear.y == pytest.approx(0.0)
        assert msg.linear.z == pytest.approx(0.0)
        assert msg.angular.x == pytest.approx(0.0)
        assert msg.angular.y == pytest.approx(0.0)
        assert msg.angular.z == pytest.approx(-0.25)
    finally:
        del subscription
        executor.shutdown()
        listener.destroy_node()
