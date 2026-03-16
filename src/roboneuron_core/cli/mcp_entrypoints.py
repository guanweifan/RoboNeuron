#!/usr/bin/env python3
"""Unified CLI entrypoints for RoboNeuron MCP services."""

from __future__ import annotations

import runpy


def _run_module(module_name: str) -> None:
    runpy.run_module(module_name, run_name="__main__")


def mcp_perception() -> None:
    _run_module("roboneuron_core.servers.perception_server")


def mcp_vla() -> None:
    _run_module("roboneuron_core.servers.vla_server")


def mcp_control() -> None:
    _run_module("roboneuron_core.servers.control_server")


def mcp_twist() -> None:
    _run_module("roboneuron_core.servers.generated.twist_server")


def mcp_eef_delta() -> None:
    _run_module("roboneuron_core.servers.generated.eef_delta_server")
