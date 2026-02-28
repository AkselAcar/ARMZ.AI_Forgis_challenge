"""Base classes for the skill system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel


@dataclass
class ExecutionContext:
    """Context passed to skills during execution."""

    flow_id: str
    step_id: str
    state_name: str
    executor_type: str = "robot"
    executors: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)

    def get_executor(self, executor_type: Optional[str] = None) -> Any:
        """Get an executor by type. Defaults to step's executor_type if not specified."""
        exec_type = executor_type or self.executor_type
        executor = self.executors.get(exec_type)
        if executor is None:
            raise ValueError(f"Executor '{exec_type}' not available in context")
        return executor

    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable from the flow context."""
        return self.variables.get(name, default)

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable in the flow context."""
        self.variables[name] = value


@dataclass
class SkillResult:
    """Result returned from skill execution."""

    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Optional[dict] = None) -> "SkillResult":
        """Create a successful result."""
        return cls(success=True, data=data or {})

    @classmethod
    def fail(cls, error: str, data: Optional[dict] = None) -> "SkillResult":
        """Create a failed result."""
        return cls(success=False, data=data or {}, error=error)


TParams = TypeVar("TParams", bound=BaseModel)


class Skill(ABC, Generic[TParams]):
    """Abstract base class for all skills."""

    # Class attributes to be defined by subclasses
    name: str = ""
    executor_type: str = "robot"
    description: str = ""

    @classmethod
    @abstractmethod
    def params_schema(cls) -> type[BaseModel]:
        """Return the Pydantic model for this skill's parameters."""
        ...

    @abstractmethod
    async def validate(self, params: TParams) -> tuple[bool, Optional[str]]:
        """
        Validate parameters before execution.

        Returns:
            Tuple of (is_valid, error_message).
            If valid, error_message is None.
        """
        ...

    @abstractmethod
    async def execute(self, params: TParams, context: ExecutionContext) -> SkillResult:
        """
        Execute the skill with the given parameters.

        Args:
            params: Validated parameters for this skill.
            context: Execution context with access to executors.

        Returns:
            SkillResult indicating success/failure and any output data.
        """
        ...

    def parse_params(self, raw_params: dict) -> TParams:
        """Parse raw dict parameters into the typed params model."""
        schema = self.params_schema()
        return schema.model_validate(raw_params)

    def get_metadata(self) -> dict:
        """Get skill metadata for API discovery."""
        schema = self.params_schema()
        return {
            "name": self.name,
            "executor_type": self.executor_type,
            "description": self.description,
            "params_schema": schema.model_json_schema(),
        }
