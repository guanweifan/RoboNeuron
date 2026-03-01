"""Typed models for simulation-domain configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    wrapper_import: str
    suite: str | dict[str, Any] | None = None
    task_id: int | None = None
    public_topic: str = "/simulation_rgb"
    input_topic: str = "/ee_command"
    rate_hz: int = 10
