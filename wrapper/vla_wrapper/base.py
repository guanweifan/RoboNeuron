from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
from PIL import Image
import torch

class ModelWrapper(ABC):
    """
    Abstract Base Class for Vision-Language-Action (VLA) Models.

    All specific VLA model implementations must inherit from this class 
    and implement the core methods: `load()` and `predict_action()`.
    """

    def __init__(self, 
                 model_path: Union[str, Path], 
                 device: Optional[torch.device] = None, 
                 **kwargs):
        """
        Initializes the ModelWrapper with path and device configuration.

        Args:
            model_path (Union[str, Path]): Path to the model files (local directory or HF hub ID).
            device (Optional[torch.device]): The device (e.g., 'cuda:0' or 'cpu') to load the model onto.
            **kwargs: Additional configuration parameters for the model or loading.
        """
        self.model_path = str(model_path)
        # Default device to cuda:0 if available, otherwise cpu
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        self.kwargs = kwargs
        self.model = None        # Placeholder for the loaded core model
        self.processor = None    # Placeholder for the loaded processor/tokenizer

    @abstractmethod
    def load(self) -> None:
        """
        Loads the model and its associated processor/tokenizer into memory and onto the device.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_action(self, 
                       image: Image.Image, 
                       instruction: str, 
                       unnorm_key: Optional[str] = None, 
                       **kwargs) -> Any:
        """
        Performs inference to predict the next robot action.

        Args:
            image (Image.Image): The input visual observation (e.g., agent view).
            instruction (str): The natural language task instruction.
            unnorm_key (Optional[str]): Key used to retrieve the correct normalization statistics 
                                        for denormalizing the predicted action.
            **kwargs: Additional parameters specific to the model's prediction API (e.g., proprioception).

        Returns:
            Any: The predicted action, which must be a serializable object 
                 (e.g., dict, list, or numpy array).
        """
        raise NotImplementedError