"""Tests for the new unified MCP CLI entrypoints."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from unittest.mock import patch

from roboneuron_core.cli import mcp_entrypoints


class TestMcpCliEntrypoints(unittest.TestCase):
    def _assert_routed(self, func: Callable[[], object], module_name: str) -> None:
        with patch("roboneuron_core.cli.mcp_entrypoints.runpy.run_module") as run_module:
            func()
            run_module.assert_called_once_with(module_name, run_name="__main__")

    def test_mcp_perception_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_perception,
            "roboneuron_core.servers.perception_server",
        )

    def test_mcp_vla_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_vla,
            "roboneuron_core.servers.vla_server",
        )

    def test_mcp_control_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_control,
            "roboneuron_core.servers.control_server",
        )

    def test_mcp_simulation_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_simulation,
            "roboneuron_core.servers.simulation_server",
        )

    def test_mcp_twist_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_twist,
            "roboneuron_core.servers.generated.twist_server",
        )

    def test_mcp_eef_delta_routes(self) -> None:
        self._assert_routed(
            mcp_entrypoints.mcp_eef_delta,
            "roboneuron_core.servers.generated.eef_delta_server",
        )


if __name__ == "__main__":
    unittest.main()
