"""Typed models for messaging-domain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopicPublishConfig:
    topic_name: str
    msg_type_name: str
