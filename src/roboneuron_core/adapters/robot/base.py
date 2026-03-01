from abc import ABC, abstractmethod
from typing import Any


class AdapterWrapper(ABC):
    """
    Abstract Base Class for Robot/Environment Adapters.

    Defines the standard interface for environment interaction: 
    observation gathering and action execution.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initializes the AdapterWrapper."""
        self.kwargs = kwargs

    @abstractmethod
    def obtain_observation(self) -> dict[str, Any]:
        """
        Gathers the current state of the environment and formats it into 
        a standardized observation dictionary for the policy.
        
        Returns:
            Dict[str, Any]: The standardized observation.
        """
        raise NotImplementedError
    
    @abstractmethod
    def step(self, action: Any) -> Any:
        """
        Translates the policy's standardized action into environment-specific 
        commands and executes one step.

        Args:
            action (Any): The standardized action output from the policy.

        Returns:
            Any: Optional post-step information (e.g., status, reward).
        """
        raise NotImplementedError
