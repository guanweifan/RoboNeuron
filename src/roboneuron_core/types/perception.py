"""Typed models for perception-domain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PerceptionConfig:
    wrapper_import: str
    topic: str = "/isaac_rgb"
    width: int = 256
    height: int = 256
    rate_hz: int = 10
