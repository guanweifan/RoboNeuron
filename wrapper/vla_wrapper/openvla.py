# wrappers/openvla.py
import json
import logging
from pathlib import Path
from typing import Optional, Union, Any
from PIL import Image
import torch

from .base import ModelWrapper

from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from vla_src.openvla.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from vla_src.openvla.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from vla_src.openvla.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Register custom components for Hugging Face Auto classes
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


logger = logging.getLogger("wrappers.openvla")


class OpenVLAWrapper(ModelWrapper):
    """
    Wrapper for OpenVLA models using the Hugging Face Transformers interface.

    Handles model loading, prompt formatting, and prediction using the custom 
    OpenVLA AutoModel components.
    """
    def __init__(self, 
                 model_path: Union[str, Path], 
                 attn_implementation: Optional[str] = "flash_attention_2", 
                 dtype: Optional[torch.dtype] = None, 
                 **kwargs):
        """
        Initializes the OpenVLA wrapper instance.

        Args:
            model_path (Union[str, Path]): Path to the model files (local directory or HF hub ID).
            attn_implementation (Optional[str]): Attention implementation mode (e.g., 'flash_attention_2').
            dtype (Optional[torch.dtype]): The torch data type to use for model loading (defaults to torch.bfloat16).
            **kwargs: Additional model loading configurations (e.g., low_cpu_mem_usage).
        """
        super().__init__(model_path, **kwargs)
        self.attn_implementation = attn_implementation
        self.torch_dtype = dtype or torch.bfloat16

    def _get_prompt(self, instruction: str) -> str:
        """
        Constructs the appropriate VLA prompt based on the model version.
        
        Args:
            instruction (str): The natural language task instruction.
            
        Returns:
            str: The formatted instruction prompt string.
        """
        if "v01" in self.model_path:
            # Chat template for older VLA versions (v01)
            return (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions. "
                f"USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
            )
        # Default prompt format for newer VLA models
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    def load(self) -> None:
        """
        Loads the OpenVLA processor and model using Hugging Face Auto classes.
        Attempts to load action normalization statistics from the local checkpoint path.
        """
        if AutoProcessor is None or AutoModelForVision2Seq is None:
            raise RuntimeError("Required transformers AutoModelForVision2Seq / AutoProcessor failed to import.")
            
        logger.info(f"Loading OpenVLA processor and model from {self.model_path}")
        
        # Load Processor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Load Model with specified configuration
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=self.kwargs.get("low_cpu_mem_usage", True),
            trust_remote_code=False,
        ).to(self.device)
        
        # Load normalization statistics for action un-normalization
        local_path = Path(self.model_path)
        if local_path.is_dir():
            stats = local_path / "dataset_statistics.json"
            if stats.exists():
                try:
                    with open(stats, "r") as f:
                        self.model.norm_stats = json.load(f)
                except Exception:
                    logger.warning("Failed to load dataset_statistics.json.")

    def predict_action(self, 
                       image: Image.Image, 
                       instruction: str, 
                       unnorm_key: Optional[str] = None, 
                       **kwargs) -> Any:
        """
        Performs inference to predict the next robot action.

        Args:
            image (Image.Image): The input visual observation.
            instruction (str): The natural language task instruction.
            unnorm_key (Optional[str]): Key for action un-normalization (optional).
            **kwargs: Additional prediction parameters.

        Returns:
            Any: The predicted action, typically a NumPy array of continuous control values.
            
        Raises:
            RuntimeError: If the model does not expose the required 'predict_action' API.
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model or processor is not loaded. Call load() first.")
            
        prompt = self._get_prompt(instruction)
        
        # Prepare inputs (image and text)
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch_dtype)
        
        # Perform inference using the model's custom method
        if not hasattr(self.model, "predict_action"):
            raise RuntimeError("Loaded OpenVLA model does not expose predict_action API.")
            
        action = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return action