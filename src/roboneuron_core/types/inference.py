"""Typed models for inference-domain configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    instruction: str
    model_path: str | None = None
    input_topic: str = "/isaac_rgb"
    output_topic: str = "/ee_command"
    accel_method: str = "none"
    accel_level: str = "off"
