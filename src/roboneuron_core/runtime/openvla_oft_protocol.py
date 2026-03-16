from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from PIL import Image

from .openvla_protocol import decode_image_from_base64, encode_image_to_base64

_IMAGE_MARKER = "__roboneuron_image__"
_IMAGE_LIST_MARKER = "__roboneuron_image_list__"


def _is_image_array(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] in {3, 4}


def _to_pil_image(value: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if not _is_image_array(value):
        raise TypeError(f"Unsupported image payload: {type(value)!r}")
    return Image.fromarray(np.asarray(value, dtype=np.uint8)).convert("RGB")


def encode_observation_for_transport(observation: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, Image.Image) or _is_image_array(value):
            payload[key] = {
                _IMAGE_MARKER: True,
                "base64": encode_image_to_base64(_to_pil_image(value)),
            }
            continue

        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            value_list = list(value)
            if value_list and all(isinstance(item, Image.Image) or _is_image_array(item) for item in value_list):
                payload[key] = {
                    _IMAGE_LIST_MARKER: True,
                    "items": [encode_image_to_base64(_to_pil_image(item)) for item in value_list],
                }
                continue

        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()
            continue

        payload[key] = value
    return payload


def decode_observation_from_transport(payload: Mapping[str, Any]) -> dict[str, Any]:
    observation: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Mapping) and value.get(_IMAGE_MARKER):
            observation[key] = decode_image_from_base64(str(value["base64"]))
            continue

        if isinstance(value, Mapping) and value.get(_IMAGE_LIST_MARKER):
            observation[key] = [decode_image_from_base64(str(item)) for item in value.get("items", [])]
            continue

        observation[key] = value
    return observation
