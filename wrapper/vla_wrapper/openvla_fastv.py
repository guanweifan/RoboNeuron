# wrappers/openvla_fastv.py
from pathlib import Path
from typing import Optional, Union, Any , Dict
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
        model_path: Union[str, Path],
        *,
        default_fastv_config: Optional[Dict[str, Any]] = None,
        attn_implementation: Optional[str] = "flash_attention_2",
        dtype: Optional[torch.dtype] = None,
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
        self.fastv_config: Optional[Dict[str, Any]] = kwargs.get("fastv_config")


    def load(self) -> None:
        """
        Loads the model and processor using the base OpenVLA logic, then sets the model to evaluation mode.
        """
        super().load()
        self.model.eval()


    def predict_action(
        self,
        image,
        instruction: str,
        unnorm_key: Optional[str] = None,
        accel_method: Optional[str] = None,
        accel_config: Optional[dict] = None,
        prune_config: Optional[dict] = None,
        fastv_config: Optional[dict] = None,
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
        prompt = self._get_prompt(instruction)
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch_dtype)

        if not hasattr(self.model, "predict_action"):
            raise RuntimeError("Loaded OpenVLA model does not expose predict_action API")

        # Determine final denormalization key (default_unnorm_key is assumed to be defined by the base class)
        final_unnorm_key = unnorm_key if unnorm_key is not None else getattr(self, "default_unnorm_key", None)

        # Logic for determining the FastV configuration
        if fastv_config is None and accel_method == "fastv":
            fastv_config = accel_config

        final_fastv_config = fastv_config if fastv_config is not None else getattr(self, "fastv_config", None)

        action = self.model.predict_action(
            **inputs,
            unnorm_key=final_unnorm_key,
            do_sample=False,
            fastv_config=final_fastv_config,
            prune_config=prune_config,  
            **kwargs,
        )
        return action