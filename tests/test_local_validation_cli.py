from __future__ import annotations

from roboneuron_core.cli.local_validation import (
    LOCAL_VALIDATION_TESTS,
    build_validation_command,
    run_local_validation,
)


def test_build_validation_command_uses_current_python() -> None:
    command = build_validation_command(python_executable="/tmp/python")

    assert command[:3] == ["/tmp/python", "-m", "pytest"]
    assert command[3] == "-q"
    assert command[4:] == list(LOCAL_VALIDATION_TESTS)


def test_run_local_validation_disables_plugin_autoload(monkeypatch, tmp_path) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "roboneuron_core.cli.local_validation._project_root",
        lambda: tmp_path,
    )

    def _fake_run(cmd, *, cwd, env, check):
        calls["cmd"] = cmd
        calls["cwd"] = cwd
        calls["env"] = env
        calls["check"] = check

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr("roboneuron_core.cli.local_validation.subprocess.run", _fake_run)

    result = run_local_validation()

    assert result == 0
    assert calls["cwd"] == tmp_path
    assert calls["check"] is False
    assert calls["env"]["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert calls["cmd"][4:] == list(LOCAL_VALIDATION_TESTS)
