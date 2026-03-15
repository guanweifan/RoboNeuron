from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .openvla import OpenVLAWrapper


class OpenVLAFastVWrapper(OpenVLAWrapper):
    """
    OpenVLA wrapper specifically designed to support FastV and other visual acceleration methods.
    
    Inherits core loading and prediction functionality from OpenVLAWrapper but 
    adds parameters for acceleration configurations.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        default_fastv_config: dict[str, Any] | None = None,
        attn_implementation: str | None = None,
        dtype: torch.dtype | str | None = None,
        **kwargs,
    ):
        """
        Initializes the OpenVLAFastV wrapper.
        
        Args:
            model_path (Union[str, Path]): Path to the model files.
            default_fastv_config (Optional[Dict[str, Any]]): Default FastV configuration dictionary.
            attn_implementation (Optional[str]): Attention implementation mode.
            dtype (Optional[torch.dtype]): The torch data type to use for model loading.
            **kwargs: Additional configuration parameters passed to the base wrapper.
        """
        # Store default FastV config if provided
        if default_fastv_config is not None:
            kwargs["fastv_config"] = default_fastv_config

        super().__init__(
            model_path=model_path,
            attn_implementation=attn_implementation,
            dtype=dtype,
            **kwargs,
        )
        # Store FastV config as an instance attribute for easy access/override
        self.fastv_config: dict[str, Any] | None = kwargs.get("fastv_config")

    def predict_action(
        self,
        image,
        instruction: str,
        unnorm_key: str | None = None,
        accel_method: str | None = None,
        accel_config: dict | None = None,
        prune_config: dict | None = None,
        fastv_config: dict | None = None,
        **kwargs,
    ) -> Any:
        """
        Performs action prediction with optional visual acceleration configuration.

        The final FastV configuration is determined by:
        1. Explicit `fastv_config` (highest precedence).
        2. `accel_config` if `accel_method` is "fastv".
        3. Instance's `default_fastv_config` (lowest precedence).

        Args:
            image (PIL.Image): The input visual observation.
            instruction (str): The natural language task instruction.
            unnorm_key (Optional[str]): Key for action denormalization (overrides default).
            accel_method (Optional[str]): Name of the desired acceleration method (e.g., "fastv").
            accel_config (Optional[dict]): Generic acceleration configuration dictionary.
            prune_config (Optional[dict]): Dictionary defining pruning parameters.
            fastv_config (Optional[dict]): Explicit FastV configuration dictionary.
            **kwargs: Additional parameters passed to the underlying model's `predict_action`.

        Returns:
            Any: The predicted action output.
        """
        final_unnorm_key = unnorm_key if unnorm_key is not None else self.default_unnorm_key

        # Logic for determining the FastV configuration
        if fastv_config is None and accel_method == "fastv":
            fastv_config = accel_config

        final_fastv_config = fastv_config if fastv_config is not None else self.fastv_config

        return self._predict_request(
            image=image,
            instruction=instruction,
            unnorm_key=final_unnorm_key,
            fastv_config=final_fastv_config,
            prune_config=prune_config,
            **kwargs,
        )
