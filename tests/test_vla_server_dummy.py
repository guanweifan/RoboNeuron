"""Unit tests for dummy-specific VLA server behavior."""

from __future__ import annotations

import importlib
import sys
import types

from roboneuron_core.adapters.vla.dummy_vla import DummyVLAWrapper
from roboneuron_core.kernel import ExecutionSessionStatus, HealthLevel


def _install_fake_ros_modules(monkeypatch) -> None:
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda *args, **kwargs: None
    fake_rclpy.ok = lambda: False
    fake_rclpy.shutdown = lambda *args, **kwargs: None
    fake_rclpy.spin = lambda *args, **kwargs: None

    fake_rclpy_node = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    fake_rclpy_node.Node = FakeNode

    fake_cv_bridge = types.ModuleType("cv_bridge")

    class FakeCvBridge:
        def imgmsg_to_cv2(self, *args, **kwargs):  # pragma: no cover - not used here
            del args, kwargs
            return None

    fake_cv_bridge.CvBridge = FakeCvBridge

    fake_sensor_msgs = types.ModuleType("sensor_msgs")
    fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class FakeRosImage:
        pass

    fake_sensor_msgs_msg.Image = FakeRosImage
    fake_sensor_msgs.msg = fake_sensor_msgs_msg

    fake_roboneuron_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_roboneuron_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")

    class FakeEEFDeltaCommand:
        def __init__(self) -> None:
            self.delta_x = 0.0
            self.delta_y = 0.0
            self.delta_z = 0.0
            self.delta_roll = 0.0
            self.delta_pitch = 0.0
            self.delta_yaw = 0.0
            self.gripper_cmd = 0.0

    class FakeRawActionChunk:
        def __init__(self) -> None:
            self.protocol = ""
            self.frame = ""
            self.action_dim = 0
            self.chunk_length = 0
            self.step_duration_sec = 0.0
            self.values = []

    class FakeTaskSpaceState:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.roll = 0.0
            self.pitch = 0.0
            self.yaw = 0.0
            self.gripper_open_fraction = 0.0

    fake_roboneuron_interfaces_msg.EEFDeltaCommand = FakeEEFDeltaCommand
    fake_roboneuron_interfaces_msg.RawActionChunk = FakeRawActionChunk
    fake_roboneuron_interfaces_msg.TaskSpaceState = FakeTaskSpaceState
    fake_roboneuron_interfaces.msg = fake_roboneuron_interfaces_msg

    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", fake_rclpy_node)
    monkeypatch.setitem(sys.modules, "cv_bridge", fake_cv_bridge)
    monkeypatch.setitem(sys.modules, "sensor_msgs", fake_sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", fake_sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_roboneuron_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_roboneuron_interfaces_msg)


def test_start_vla_inference_allows_dummy_without_model_path(monkeypatch) -> None:
    _install_fake_ros_modules(monkeypatch)
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    monkeypatch.setattr(vla_server, "get_registry", lambda: {"dummy": DummyVLAWrapper})
    monkeypatch.setattr(
        vla_server,
        "_load_vla_models_config",
        lambda: (_ for _ in ()).throw(AssertionError("dummy should not read model config")),
    )

    class FakeProcess:
        def __init__(self, target=None, args=(), daemon=False) -> None:
            del target, args, daemon
            self.pid = 4321
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

    monkeypatch.setattr(vla_server.multiprocessing, "get_context", lambda method: FakeContext())
    vla_server._VLA_PROCESS = None

    result = vla_server.start_vla_inference(
        model_name="dummy",
        model_path=None,
        instruction="test task",
    )

    assert result == "Success: VLA dummy started (pid=4321)."
    assert vla_server._VLA_SESSION is not None
    assert vla_server._VLA_SESSION.status == ExecutionSessionStatus.RUNNING
    assert vla_server._VLA_SESSION.runtime_profile is not None
    assert vla_server._VLA_SESSION.runtime_profile.layer == "core"
    assert vla_server._VLA_SESSION.trace is not None
    assert vla_server._VLA_SESSION.trace.profile_name == "dummy"
    assert vla_server._VLA_HEALTH.level == HealthLevel.READY

    stop_result = vla_server.stop_vla_inference()

    assert stop_result == "Success: VLA stopped."
    assert vla_server._VLA_SESSION.status == ExecutionSessionStatus.STOPPED
    assert vla_server._VLA_SESSION.history[-1].details["requested_by"] == "mcp"
    assert vla_server._VLA_HEALTH.level == HealthLevel.IDLE
