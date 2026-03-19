from __future__ import annotations

import importlib
import sys
import types


def _install_fake_ros_modules() -> None:
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda *args, **kwargs: None
    fake_rclpy.shutdown = lambda *args, **kwargs: None
    fake_rclpy.spin = lambda *args, **kwargs: None

    fake_rclpy_node = types.ModuleType("rclpy.node")

    class FakeNode:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

    fake_rclpy_node.Node = FakeNode

    fake_cv_bridge = types.ModuleType("cv_bridge")

    class FakeCvBridge:
        def cv2_to_imgmsg(self, *args, **kwargs):  # pragma: no cover - not used here
            del args, kwargs
            return None

    fake_cv_bridge.CvBridge = FakeCvBridge

    fake_sensor_msgs = types.ModuleType("sensor_msgs")
    fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")

    class FakeRosImage:
        pass

    fake_sensor_msgs_msg.Image = FakeRosImage
    fake_sensor_msgs.msg = fake_sensor_msgs_msg

    sys.modules["rclpy"] = fake_rclpy
    sys.modules["rclpy.node"] = fake_rclpy_node
    sys.modules["cv_bridge"] = fake_cv_bridge
    sys.modules["sensor_msgs"] = fake_sensor_msgs
    sys.modules["sensor_msgs.msg"] = fake_sensor_msgs_msg


def test_start_camera_allows_multiple_topics(monkeypatch) -> None:
    _install_fake_ros_modules()
    module_name = "roboneuron_core.servers.perception_server"
    sys.modules.pop(module_name, None)
    perception_server = importlib.import_module(module_name)

    class FakeProcess:
        _next_pid = 1000

        def __init__(self, target=None, args=(), daemon=False) -> None:
            del target, args, daemon
            type(self)._next_pid += 1
            self.pid = type(self)._next_pid
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

    monkeypatch.setattr(perception_server.multiprocessing, "get_context", lambda method: FakeContext())
    monkeypatch.setattr(perception_server.importlib, "import_module", lambda module_name: object())
    perception_server._CAMERA_PROCESSES.clear()

    first = perception_server.start_camera("roboneuron_core.adapters.camera.dummy_camera.DummyCameraWrapper", "/cam0")
    second = perception_server.start_camera("roboneuron_core.adapters.camera.dummy_camera.DummyCameraWrapper", "/cam1")
    duplicate = perception_server.start_camera("roboneuron_core.adapters.camera.dummy_camera.DummyCameraWrapper", "/cam0")

    assert "Success: Camera started publishing to /cam0" in first
    assert "Success: Camera started publishing to /cam1" in second
    assert duplicate == "Error: Camera is already running on /cam0. Call stop_camera(topic=...) first."
    assert sorted(perception_server._CAMERA_PROCESSES.keys()) == ["/cam0", "/cam1"]


def test_stop_camera_can_stop_single_topic_or_all(monkeypatch) -> None:
    _install_fake_ros_modules()
    module_name = "roboneuron_core.servers.perception_server"
    sys.modules.pop(module_name, None)
    perception_server = importlib.import_module(module_name)

    class FakeProcess:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            del timeout

        def kill(self) -> None:
            self._alive = False

    perception_server._CAMERA_PROCESSES.clear()
    perception_server._CAMERA_PROCESSES["/cam0"] = FakeProcess()
    perception_server._CAMERA_PROCESSES["/cam1"] = FakeProcess()

    single = perception_server.stop_camera("/cam0")
    assert single == "Success: Camera on /cam0 stopped."
    assert sorted(perception_server._CAMERA_PROCESSES.keys()) == ["/cam1"]

    all_result = perception_server.stop_camera()
    assert all_result == "Success: Camera on /cam1 stopped."
    assert perception_server._CAMERA_PROCESSES == {}
