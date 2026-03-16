from abc import ABC, abstractmethod
from typing import Any


class AdapterWrapper(ABC):
    """Base interface for robot-side adapters."""

    def __init__(self, **kwargs: Any) -> None:
        """Store adapter-specific options."""
        self.kwargs = kwargs

    @abstractmethod
    def obtain_observation(self) -> dict[str, Any]:
        """Return the current observation in the policy-facing format."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Any:
        """Apply one action and return any adapter-specific result."""
        raise NotImplementedError
