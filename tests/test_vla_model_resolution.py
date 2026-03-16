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
    fake_rclpy_node.Node = object

    fake_cv_bridge = types.ModuleType("cv_bridge")
    fake_cv_bridge.CvBridge = object

    fake_sensor_msgs = types.ModuleType("sensor_msgs")
    fake_sensor_msgs_msg = types.ModuleType("sensor_msgs.msg")
    fake_sensor_msgs_msg.Image = object
    fake_sensor_msgs.msg = fake_sensor_msgs_msg

    fake_roboneuron_interfaces = types.ModuleType("roboneuron_interfaces")
    fake_roboneuron_interfaces_msg = types.ModuleType("roboneuron_interfaces.msg")
    fake_roboneuron_interfaces_msg.EEFDeltaCommand = object
    fake_roboneuron_interfaces.msg = fake_roboneuron_interfaces_msg

    sys.modules["rclpy"] = fake_rclpy
    sys.modules["rclpy.node"] = fake_rclpy_node
    sys.modules["cv_bridge"] = fake_cv_bridge
    sys.modules["sensor_msgs"] = fake_sensor_msgs
    sys.modules["sensor_msgs.msg"] = fake_sensor_msgs_msg
    sys.modules["roboneuron_interfaces"] = fake_roboneuron_interfaces
    sys.modules["roboneuron_interfaces.msg"] = fake_roboneuron_interfaces_msg


def test_resolve_model_spec_supports_runtime_kwargs(monkeypatch) -> None:
    _install_fake_ros_modules()
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    monkeypatch.setattr(
        vla_server,
        "_load_vla_models_config",
        lambda: {
            "openvla": {
                "path": "checkpoints/openvla/openvla-7b",
                "kwargs": {
                    "runtime_python": ".venvs/openvla/bin/python",
                    "default_unnorm_key": "bridge_orig",
                },
            }
        },
    )

    model_path, model_kwargs = vla_server._resolve_model_spec("openvla", None)

    assert model_path == "checkpoints/openvla/openvla-7b"
    assert model_kwargs == {
        "runtime_python": ".venvs/openvla/bin/python",
        "default_unnorm_key": "bridge_orig",
    }


def test_resolve_model_spec_supports_openvla_oft_runtime_kwargs(monkeypatch) -> None:
    _install_fake_ros_modules()
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    monkeypatch.setattr(
        vla_server,
        "_load_vla_models_config",
        lambda: {
            "openvla-oft": {
                "path": "checkpoints/openvla-oft/openvla-oft-pick-banana",
                "kwargs": {
                    "runtime_python": ".venvs/openvla-oft/bin/python",
                    "default_unnorm_key": "vr_banana",
                    "robot_platform": "bridge",
                    "use_proprio": True,
                },
            }
        },
    )

    model_path, model_kwargs = vla_server._resolve_model_spec("openvla-oft", None)

    assert model_path == "checkpoints/openvla-oft/openvla-oft-pick-banana"
    assert model_kwargs == {
        "runtime_python": ".venvs/openvla-oft/bin/python",
        "default_unnorm_key": "vr_banana",
        "robot_platform": "bridge",
        "use_proprio": True,
    }
