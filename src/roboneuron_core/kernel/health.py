"""Kernel-level runtime health primitives."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class HealthLevel(StrEnum):
    UNKNOWN = "unknown"
    IDLE = "idle"
    READY = "ready"
    DEGRADED = "degraded"
    ERROR = "error"


@dataclass(frozen=True)
class HealthStatus:
    """Minimal health snapshot for a runtime component."""

    component: str
    level: HealthLevel
    summary: str
    checked_at_sec: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def idle(
        cls,
        component: str,
        *,
        summary: str = "runtime is idle",
        details: dict[str, Any] | None = None,
        checked_at_sec: float | None = None,
    ) -> HealthStatus:
        return cls(
            component=component,
            level=HealthLevel.IDLE,
            summary=summary,
            checked_at_sec=time.time() if checked_at_sec is None else checked_at_sec,
            details=dict(details or {}),
        )

    @classmethod
    def ready(
        cls,
        component: str,
        *,
        summary: str = "runtime is ready",
        details: dict[str, Any] | None = None,
        checked_at_sec: float | None = None,
    ) -> HealthStatus:
        return cls(
            component=component,
            level=HealthLevel.READY,
            summary=summary,
            checked_at_sec=time.time() if checked_at_sec is None else checked_at_sec,
            details=dict(details or {}),
        )

    @classmethod
    def error(
        cls,
        component: str,
        *,
        summary: str,
        details: dict[str, Any] | None = None,
        checked_at_sec: float | None = None,
    ) -> HealthStatus:
        return cls(
            component=component,
            level=HealthLevel.ERROR,
            summary=summary,
            checked_at_sec=time.time() if checked_at_sec is None else checked_at_sec,
            details=dict(details or {}),
        )

    @property
    def is_healthy(self) -> bool:
        return self.level in {HealthLevel.IDLE, HealthLevel.READY}

