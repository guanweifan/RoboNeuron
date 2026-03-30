"""Runtime profile primitives for the current RoboNeuron execution path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_RUNTIME_LAYERS = {"core", "edge"}
_DEPLOYMENT_MODES = {"local", "hybrid"}


@dataclass(frozen=True)
class RuntimeProfile:
    """Minimal runtime profile aligned with the current control and VLA path."""

    name: str
    layer: str
    deployment_mode: str
    action_transport: str | None = None
    action_protocol: str | None = None
    state_source: str | None = None
    model_runtime: str | None = None
    robot_backend: str | None = None
    vendor_stack: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layer not in _RUNTIME_LAYERS:
            raise ValueError(f"layer must be one of {_RUNTIME_LAYERS}, got {self.layer!r}.")
        if self.deployment_mode not in _DEPLOYMENT_MODES:
            raise ValueError(
                f"deployment_mode must be one of {_DEPLOYMENT_MODES}, got {self.deployment_mode!r}."
            )
        object.__setattr__(self, "vendor_stack", tuple(str(item) for item in self.vendor_stack))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def core_vla(
        cls,
        *,
        name: str,
        deployment_mode: str,
        model_runtime: str,
        action_transport: str,
        action_protocol: str | None = None,
        state_source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeProfile:
        return cls(
            name=name,
            layer="core",
            deployment_mode=deployment_mode,
            model_runtime=model_runtime,
            action_transport=action_transport,
            action_protocol=action_protocol,
            state_source=state_source,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def edge_control(
        cls,
        *,
        name: str,
        deployment_mode: str,
        robot_backend: str | None,
        action_transport: str,
        action_protocol: str | None = None,
        state_source: str | None = None,
        vendor_stack: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeProfile:
        return cls(
            name=name,
            layer="edge",
            deployment_mode=deployment_mode,
            robot_backend=robot_backend,
            action_transport=action_transport,
            action_protocol=action_protocol,
            state_source=state_source,
            vendor_stack=vendor_stack,
            metadata=dict(metadata or {}),
        )
