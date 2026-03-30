"""Helpers for chunked raw action ROS messages."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from roboneuron_core.kernel import ActionChunk, ActionContract, RawActionStep

RAW_ACTION_CHUNK_TOPIC = "/raw_action_chunk"


def _raw_action_chunk_cls():
    from roboneuron_interfaces.msg import RawActionChunk

    return RawActionChunk


def array_to_raw_action_chunk_message(
    actions: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    protocol: str,
    step_duration_sec: float = 0.1,
    frame: str = "tool",
) -> object:
    """Convert raw action data into a ``RawActionChunk`` ROS message."""

    contract = ActionContract.raw_action_chunk(protocol=protocol, frame=frame)
    action_np = contract.validate_action_matrix(actions)
    if step_duration_sec <= 0:
        raise ValueError("step_duration_sec must be positive.")

    message = _raw_action_chunk_cls()()
    message.protocol = protocol
    message.frame = frame
    message.action_dim = int(action_np.shape[1])
    message.chunk_length = int(action_np.shape[0])
    message.step_duration_sec = float(step_duration_sec)
    message.values = action_np.reshape(-1).tolist()
    return message


def raw_action_chunk_message_to_action_chunk(message: object) -> ActionChunk:
    """Convert a ``RawActionChunk`` ROS message into a semantic action chunk."""

    action_dim = int(message.action_dim)
    if action_dim <= 0:
        raise ValueError(f"RawActionChunk.action_dim must be positive, got {action_dim}.")

    values = np.asarray(message.values, dtype=np.float64).reshape(-1)
    if values.size % action_dim != 0:
        raise ValueError(
            f"Raw action payload length {values.size} is not divisible by action_dim {action_dim}."
        )

    chunk_length = int(getattr(message, "chunk_length", 0)) or (values.size // action_dim)
    protocol = str(getattr(message, "protocol", "") or "eef_delta")
    frame = str(getattr(message, "frame", "") or "tool")
    step_duration_sec = float(getattr(message, "step_duration_sec", 0.1))
    contract = ActionContract.raw_action_chunk(
        protocol=protocol,
        frame=frame,
        action_dim=action_dim,
    )
    action_matrix = contract.validate_action_matrix(values.reshape(chunk_length, action_dim))

    return ActionChunk(
        steps=tuple(RawActionStep(step, protocol=protocol, frame=frame) for step in action_matrix),
        step_duration_sec=step_duration_sec,
    )
