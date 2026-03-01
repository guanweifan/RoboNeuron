# wrappers/dummy.py
from typing import Any, Optional
from PIL import Image
from .base import ModelWrapper
import numpy as np

# Define a constant dummy action output (e.g., 7-dimensional joint space delta)
DUMMY_ACTION = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

class DummyVLAWrapper(ModelWrapper):
    """
    Placeholder wrapper for development and testing.

    This class implements the ModelWrapper interface but avoids loading 
    any actual model components, returning a constant action for all inputs. 
    Ideal for debugging integration pipelines.
    """
    
    def load(self) -> None:
        """
        Placeholder implementation. No actual model or processor is loaded.
        """
        self.model = True 
        self.processor = True
        return

    def predict_action(self, 
                       image: Image.Image, 
                       instruction: str, 
                       unnorm_key: Optional[str] = None, 
                       **kwargs) -> Any:
        """
        Returns a fixed, pre-defined dummy action regardless of input image or instruction.

        Args:
            image (Image.Image): Input visual observation (ignored).
            instruction (str): Task instruction (ignored).
            unnorm_key (Optional[str]): Denormalization key (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            Any: A fixed NumPy array representing the dummy action.
        """
        return DUMMY_ACTION