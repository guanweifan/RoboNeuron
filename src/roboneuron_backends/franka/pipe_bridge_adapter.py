"""ROS 2 adapter that forwards JointTrajectory targets into the local libfranka pipe bridge."""

from __future__ import annotations

import queue
import subprocess
import threading
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory


def _project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError("Could not locate project root containing pyproject.toml.")


def _default_bridge_executable() -> str:
    return str(
        (
            _project_root()
            / "ros"
            / "install"
            / "roboneuron_franka_bridge"
            / "lib"
            / "roboneuron_franka_bridge"
            / "franka_joint_impedance_pipe_bridge"
        ).resolve()
    )


class FrankaPipeBridgeAdapter(Node):
    """Bridge RoboNeuron's ROS 2 joint target topic to the local libfranka subprocess."""

    def __init__(self) -> None:
        super().__init__("franka_pipe_bridge_adapter")

        self._robot_ip = self.declare_parameter("robot_ip", "").value
        self._command_topic = self.declare_parameter(
            "command_topic",
            "/fr3_arm_controller/joint_trajectory",
        ).value
        self._joint_state_topic = self.declare_parameter("joint_state_topic", "/joint_states").value
        self._bridge_state_rate_hz = float(
            self.declare_parameter("bridge_state_rate_hz", 100.0).value
        )
        self._bridge_executable = self.declare_parameter(
            "bridge_executable",
            _default_bridge_executable(),
        ).value
        self._joint_names = list(
            self.declare_parameter(
                "joint_names",
                [
                    "fr3_joint1",
                    "fr3_joint2",
                    "fr3_joint3",
                    "fr3_joint4",
                    "fr3_joint5",
                    "fr3_joint6",
                    "fr3_joint7",
                ],
            ).value
        )

        if not self._robot_ip:
            raise ValueError("robot_ip parameter must not be empty.")
        if len(self._joint_names) != 7:
            raise ValueError("joint_names must contain exactly 7 arm joints.")

        self._joint_name_to_index = {name: index for index, name in enumerate(self._joint_names)}
        self._state_queue: queue.SimpleQueue[tuple[list[float], list[float]]] = queue.SimpleQueue()
        self._last_arm_positions: list[float] | None = None
        self._process_closed = False
        self._first_state_logged = False
        self._first_command_logged = False

        self._joint_state_pub = self.create_publisher(JointState, self._joint_state_topic, 10)
        self.create_subscription(JointTrajectory, self._command_topic, self._command_cb, 10)
        self.create_timer(0.01, self._drain_state_queue)
        self.create_timer(0.5, self._check_bridge_process)

        self._start_bridge_process()

    def close(self) -> None:
        if self._process_closed:
            return
        self._process_closed = True
        if self._bridge.stdin is not None:
            try:
                self._bridge.stdin.write("QUIT\n")
                self._bridge.stdin.flush()
            except Exception:
                pass
        try:
            self._bridge.terminate()
        except Exception:
            pass
        try:
            self._bridge.wait(timeout=2.0)
        except Exception:
            try:
                self._bridge.kill()
            except Exception:
                pass

    def _start_bridge_process(self) -> None:
        cmd = [
            self._bridge_executable,
            "--robot-ip",
            self._robot_ip,
            "--state-rate-hz",
            str(self._bridge_state_rate_hz),
        ]
        self.get_logger().info(f"Starting Franka pipe bridge: {' '.join(cmd)}")
        self._bridge = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        self._stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _stdout_loop(self) -> None:
        assert self._bridge.stdout is not None
        for raw_line in self._bridge.stdout:
            line = raw_line.strip()
            if not line or not line.startswith("STATE "):
                continue
            parts = line.split()
            if len(parts) != 15:
                continue
            try:
                values = [float(part) for part in parts[1:]]
            except ValueError:
                continue
            q = values[:7]
            dq = values[7:]
            self._state_queue.put((q, dq))

    def _stderr_loop(self) -> None:
        assert self._bridge.stderr is not None
        for raw_line in self._bridge.stderr:
            line = raw_line.strip()
            if not line:
                continue
            self.get_logger().info(f"[bridge] {line}")

    def _drain_state_queue(self) -> None:
        latest: tuple[list[float], list[float]] | None = None
        while True:
            try:
                latest = self._state_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        q, dq = latest
        self._last_arm_positions = q
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(self._joint_names)
        msg.position = q
        msg.velocity = dq
        self._joint_state_pub.publish(msg)
        if not self._first_state_logged:
            self.get_logger().info("Published first arm JointState from Franka pipe bridge.")
            self._first_state_logged = True

    def _check_bridge_process(self) -> None:
        code = self._bridge.poll()
        if code is None:
            return
        raise RuntimeError(f"Franka pipe bridge exited unexpectedly with code {code}.")

    def _command_cb(self, msg: JointTrajectory) -> None:
        if not msg.points:
            return

        point = msg.points[0]
        if not point.positions:
            return

        target = list(self._last_arm_positions or [0.0] * len(self._joint_names))
        if not msg.joint_names:
            if len(point.positions) < len(target):
                return
            target = [float(value) for value in point.positions[: len(target)]]
        else:
            saw_arm_joint = False
            for name, position in zip(msg.joint_names, point.positions, strict=False):
                joint_index = self._joint_name_to_index.get(name)
                if joint_index is None:
                    continue
                target[joint_index] = float(position)
                saw_arm_joint = True
            if not saw_arm_joint:
                return

        if self._bridge.stdin is None:
            raise RuntimeError("Bridge stdin is unavailable.")
        self._bridge.stdin.write("SET " + " ".join(f"{value:.17g}" for value in target) + "\n")
        self._bridge.stdin.flush()
        if not self._first_command_logged:
            self.get_logger().info("Forwarded first arm joint target into Franka pipe bridge.")
            self._first_command_logged = True


def main() -> None:
    rclpy.init()
    node = FrankaPipeBridgeAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
