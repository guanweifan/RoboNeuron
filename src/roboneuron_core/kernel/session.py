"""Kernel-level execution session lifecycle primitives."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .contracts import ActionContract
from .profile import RuntimeProfile


class ExecutionSessionStatus(StrEnum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass(frozen=True)
class ExecutionSessionEvent:
    status: ExecutionSessionStatus
    changed_at_sec: float
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionTrace:
    """Minimal execution trace aligned with the current canonical path."""

    session_id: str
    owner: str
    profile_name: str
    runtime_layer: str
    deployment_mode: str
    recorded_at_sec: float
    action_transport: str | None = None
    action_protocol: str | None = None
    state_source: str | None = None
    instruction: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(
        cls,
        session: ExecutionSession,
        *,
        runtime_profile: RuntimeProfile,
        recorded_at_sec: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionTrace:
        action_transport = (
            session.action_contract.transport
            if session.action_contract is not None
            else runtime_profile.action_transport
        )
        action_protocol = (
            session.action_contract.protocol
            if session.action_contract is not None
            else runtime_profile.action_protocol
        )
        merged_metadata = dict(runtime_profile.metadata)
        merged_metadata.update(metadata or {})
        return cls(
            session_id=session.session_id,
            owner=session.owner,
            profile_name=runtime_profile.name,
            runtime_layer=runtime_profile.layer,
            deployment_mode=runtime_profile.deployment_mode,
            recorded_at_sec=time.time() if recorded_at_sec is None else recorded_at_sec,
            action_transport=action_transport,
            action_protocol=action_protocol,
            state_source=runtime_profile.state_source,
            instruction=session.instruction,
            metadata=merged_metadata,
        )


@dataclass
class ExecutionSession:
    """Minimal lifecycle owner for the current execution path."""

    session_id: str
    owner: str
    action_contract: ActionContract | None = None
    runtime_profile: RuntimeProfile | None = None
    trace: ExecutionTrace | None = None
    instruction: str | None = None
    status: ExecutionSessionStatus = ExecutionSessionStatus.CREATED
    created_at_sec: float = field(default_factory=time.time)
    started_at_sec: float | None = None
    finished_at_sec: float | None = None
    last_updated_at_sec: float = field(default_factory=time.time)
    failure_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    history: list[ExecutionSessionEvent] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        owner: str,
        action_contract: ActionContract | None = None,
        runtime_profile: RuntimeProfile | None = None,
        instruction: str | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        now: float | None = None,
    ) -> ExecutionSession:
        created_at_sec = time.time() if now is None else now
        session = cls(
            session_id=session_id or uuid.uuid4().hex,
            owner=owner,
            action_contract=action_contract,
            runtime_profile=runtime_profile,
            instruction=instruction,
            created_at_sec=created_at_sec,
            last_updated_at_sec=created_at_sec,
            metadata=dict(metadata or {}),
        )
        if runtime_profile is not None:
            session.trace = ExecutionTrace.from_session(
                session,
                runtime_profile=runtime_profile,
                recorded_at_sec=created_at_sec,
            )
        session.history.append(
            ExecutionSessionEvent(
                status=ExecutionSessionStatus.CREATED,
                changed_at_sec=created_at_sec,
                details=session._build_event_details(),
            )
        )
        return session

    def _build_event_details(self, details: dict[str, Any] | None = None) -> dict[str, Any]:
        event_details = dict(details or {})
        if self.action_contract is not None:
            event_details.setdefault("action_transport", self.action_contract.transport)
            event_details.setdefault("action_protocol", self.action_contract.protocol)
            event_details.setdefault("action_frame", self.action_contract.frame)
        if self.runtime_profile is not None:
            event_details.setdefault("profile_name", self.runtime_profile.name)
            event_details.setdefault("runtime_layer", self.runtime_profile.layer)
            event_details.setdefault("deployment_mode", self.runtime_profile.deployment_mode)
        return event_details

    def _transition(
        self,
        status: ExecutionSessionStatus,
        *,
        now: float | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        changed_at_sec = time.time() if now is None else now
        self.status = status
        self.last_updated_at_sec = changed_at_sec
        self.history.append(
            ExecutionSessionEvent(
                status=status,
                changed_at_sec=changed_at_sec,
                reason=reason,
                details=self._build_event_details(details),
            )
        )

    def mark_starting(self, *, now: float | None = None, details: dict[str, Any] | None = None) -> None:
        self._transition(ExecutionSessionStatus.STARTING, now=now, details=details)

    def mark_running(self, *, now: float | None = None, details: dict[str, Any] | None = None) -> None:
        changed_at_sec = time.time() if now is None else now
        if self.started_at_sec is None:
            self.started_at_sec = changed_at_sec
        self._transition(ExecutionSessionStatus.RUNNING, now=changed_at_sec, details=details)

    def mark_stopped(
        self,
        *,
        now: float | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        changed_at_sec = time.time() if now is None else now
        self.finished_at_sec = changed_at_sec
        self._transition(
            ExecutionSessionStatus.STOPPED,
            now=changed_at_sec,
            reason=reason,
            details=details,
        )

    def mark_failed(
        self,
        reason: str,
        *,
        now: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        changed_at_sec = time.time() if now is None else now
        self.failure_reason = reason
        self.finished_at_sec = changed_at_sec
        self._transition(
            ExecutionSessionStatus.FAILED,
            now=changed_at_sec,
            reason=reason,
            details=details,
        )
