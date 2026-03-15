from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def build_openvla_prompt(instruction: str, model_path: str | Path) -> str:
    """Match OpenVLA's prompt format for both v0.1 and newer checkpoints."""
    model_path_str = str(model_path)
    if "v01" in model_path_str:
        return (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions. "
            f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def encode_image_to_base64(image: Image.Image) -> str:
    """Serialize a PIL image as base64-encoded PNG bytes."""
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_image_from_base64(payload: str) -> Image.Image:
    """Deserialize a base64-encoded PNG payload into a PIL image."""
    return Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")


def to_jsonable_action(action: Any) -> Any:
    """Convert common tensor/array outputs into JSON-serializable types."""
    if isinstance(action, np.ndarray):
        return action.tolist()
    if hasattr(action, "detach"):
        action = action.detach()
    if hasattr(action, "cpu"):
        action = action.cpu()
    if hasattr(action, "numpy"):
        return action.numpy().tolist()
    if hasattr(action, "tolist"):
        return action.tolist()
    return action

