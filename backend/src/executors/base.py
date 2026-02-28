"""Base class for executors."""

from abc import ABC, abstractmethod
from typing import Any


class Executor(ABC):
    """
    Abstract base class for executors.

    Executors bridge between skills and underlying hardware/ROS 2 nodes.
    Each executor type handles a specific category of operations.
    """

    executor_type: str = ""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the executor (connect to nodes, verify state)."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Cleanup executor resources."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """Check if executor is ready for operations."""
        ...

    def get_type(self) -> str:
        """Return the executor type identifier."""
        return self.executor_type
