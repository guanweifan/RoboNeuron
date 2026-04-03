from __future__ import annotations

import importlib
import sys
import types


def _install_fake_ros_modules(monkeypatch) -> None:
    fake_rclpy = types.ModuleType("rclpy")
    fake_rclpy.init = lambda *args, **kwargs: None
    fake_rclpy.ok = lambda: False
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
    fake_roboneuron_interfaces_msg.RawActionChunk = object
    fake_roboneuron_interfaces_msg.TaskSpaceState = object
    fake_roboneuron_interfaces.msg = fake_roboneuron_interfaces_msg

    monkeypatch.setitem(sys.modules, "rclpy", fake_rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.node", fake_rclpy_node)
    monkeypatch.setitem(sys.modules, "cv_bridge", fake_cv_bridge)
    monkeypatch.setitem(sys.modules, "sensor_msgs", fake_sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", fake_sensor_msgs_msg)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces", fake_roboneuron_interfaces)
    monkeypatch.setitem(sys.modules, "roboneuron_interfaces.msg", fake_roboneuron_interfaces_msg)


def test_resolve_model_spec_supports_runtime_kwargs(monkeypatch) -> None:
    _install_fake_ros_modules(monkeypatch)
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
                    "runtime_quantization": "4bit",
                    "default_unnorm_key": "bridge_orig",
                },
            }
        },
    )

    model_path, model_kwargs = vla_server._resolve_model_spec("openvla", None)

    assert model_path == "checkpoints/openvla/openvla-7b"
    assert model_kwargs == {
        "runtime_python": ".venvs/openvla/bin/python",
        "runtime_quantization": "4bit",
        "default_unnorm_key": "bridge_orig",
    }


def test_resolve_model_spec_supports_openvla_oft_runtime_kwargs(monkeypatch) -> None:
    _install_fake_ros_modules(monkeypatch)
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
                    "runtime_quantization": "8bit",
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
        "runtime_quantization": "8bit",
        "default_unnorm_key": "vr_banana",
        "robot_platform": "bridge",
        "use_proprio": True,
    }


def test_resolve_output_contract_defaults_openvla_oft_to_raw_chunks(monkeypatch) -> None:
    _install_fake_ros_modules(monkeypatch)
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    output_mode, action_protocol, action_frame = vla_server._resolve_output_contract(
        "openvla-oft",
        "auto",
        None,
        "tool",
    )

    assert output_mode == "raw_action_chunk"
    assert action_protocol == "normalized_cartesian_velocity"
    assert action_frame == "tool"


def test_resolve_output_topic_switches_default_chunk_topic(monkeypatch) -> None:
    _install_fake_ros_modules(monkeypatch)
    module_name = "roboneuron_core.servers.vla_server"
    sys.modules.pop(module_name, None)
    sys.modules.pop("roboneuron_core.utils.eef_delta", None)
    vla_server = importlib.import_module(module_name)

    resolved = vla_server._resolve_output_topic("/eef_delta_cmd", "raw_action_chunk")

    assert resolved == "/raw_action_chunk"
