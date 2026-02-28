"""Flow schema definitions for the flowdiagram execution system."""

from enum import Enum
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from executors import Executor


class ErrorHandling(str, Enum):
    """Error handling strategies for skill execution."""

    STOP = "stop"  # Abort entire flow
    RETRY = "retry"  # Retry N times, then stop
    SKIP = "skip"  # Log error, continue to next step
    FALLBACK = "fallback"  # Execute fallback skill


class ErrorConfig(BaseModel):
    """Configuration for error handling on a per-step basis."""

    strategy: ErrorHandling = ErrorHandling.STOP
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_ms: int = Field(default=1000, ge=0, le=60000)
    fallback_skill: Optional[str] = None


class StepSchema(BaseModel):
    """A single step within a state, executing a skill."""

    id: str = Field(..., min_length=1, description="Unique step identifier")
    skill: str = Field(..., min_length=1, description="Skill name to execute")
    executor: Literal["robot", "camera", "io_robot", "hand"] = Field(
        default="robot", description="Executor type for this step"
    )
    params: dict = Field(default_factory=dict, description="Skill parameters")
    store_result: Optional[str] = Field(
        default=None, description="Variable name to store the skill result (stores result.data)"
    )
    error_handling: ErrorConfig = Field(
        default_factory=ErrorConfig, description="Error handling configuration"
    )
    timeout_ms: int = Field(
        default=30000, ge=100, le=600000, description="Timeout in milliseconds"
    )


class StateSchema(BaseModel):
    """A state in the flow containing zero or more steps."""

    name: str = Field(..., min_length=1, description="State name (unique within flow)")
    steps: list[StepSchema] = Field(
        default_factory=list, description="Steps to execute in this state"
    )


class TransitionSchema(BaseModel):
    """Transition between states."""

    type: Literal["sequential", "conditional"] = Field(
        default="sequential", description="Transition type"
    )
    from_state: str = Field(..., min_length=1, description="Source state name")
    to_state: str = Field(..., min_length=1, description="Target state name")
    condition: Optional[str] = Field(
        default=None, description="Condition expression for conditional transitions"
    )


class FlowSchema(BaseModel):
    """Complete flow definition."""

    id: str = Field(..., min_length=1, description="Unique flow identifier")
    name: str = Field(..., min_length=1, description="Human-readable flow name")
    initial_state: str = Field(..., min_length=1, description="Starting state name")
    loop: bool = Field(
        default=False,
        description="If True, restart from initial_state after last state completes",
    )
    variables: dict = Field(
        default_factory=dict, description="Flow variables and lookup tables"
    )
    states: list[StateSchema] = Field(..., min_length=1, description="List of states")
    transitions: list[TransitionSchema] = Field(
        default_factory=list, description="State transitions"
    )

    def get_state(self, name: str) -> Optional[StateSchema]:
        """Get a state by name."""
        for state in self.states:
            if state.name == name:
                return state
        return None

    def has_conditional_transitions(self, state_name: str) -> bool:
        """Check if a state has any conditional transitions."""
        return any(
            t.from_state == state_name and t.type == "conditional"
            for t in self.transitions
        )

    def get_sequential_transition(self, current_state: str) -> Optional[str]:
        """Get the sequential transition target from a state (if any)."""
        for transition in self.transitions:
            if transition.from_state == current_state and transition.type == "sequential":
                return transition.to_state
        return None

    def evaluate_conditional_transitions(
        self,
        current_state: str,
        variables: Optional[dict[str, Any]] = None,
        executors: Optional[dict[str, "Executor"]] = None,
    ) -> Optional[str]:
        """
        Evaluate conditional transitions from a state.

        Args:
            current_state: Current state name
            variables: Flow runtime variables
            executors: Dict of executors (for IO-based conditions)

        Returns:
            Target state name if a condition matches, None otherwise
        """
        from .conditions import evaluate_condition

        variables = variables or {}
        executors = executors or {}

        for transition in self.transitions:
            if (
                transition.from_state == current_state
                and transition.type == "conditional"
                and transition.condition
            ):
                if evaluate_condition(transition.condition, variables, executors):
                    return transition.to_state

        return None

    def get_next_state(
        self,
        current_state: str,
        variables: Optional[dict[str, Any]] = None,
        executors: Optional[dict[str, "Executor"]] = None,
    ) -> Optional[str]:
        """
        Get the next state based on transitions.

        Evaluates conditional transitions first (in order), then falls back
        to sequential transitions if no conditional matches.

        Args:
            current_state: Current state name
            variables: Flow runtime variables (for condition evaluation)
            executors: Dict of executors (for IO-based conditions)

        Returns:
            Next state name or None if no transition found
        """
        # First, check conditional transitions
        result = self.evaluate_conditional_transitions(current_state, variables, executors)
        if result:
            return result

        # Fall back to sequential transition
        return self.get_sequential_transition(current_state)

    def validate_flow(self) -> tuple[bool, Optional[str]]:
        """Validate flow structure."""
        state_names = {s.name for s in self.states}

        # Check initial state exists
        if self.initial_state not in state_names:
            return False, f"Initial state '{self.initial_state}' not found in states"

        # Check all transitions reference valid states
        for t in self.transitions:
            if t.from_state not in state_names:
                return False, f"Transition from_state '{t.from_state}' not found"
            if t.to_state not in state_names:
                return False, f"Transition to_state '{t.to_state}' not found"

        # Check for duplicate state names
        if len(state_names) != len(self.states):
            return False, "Duplicate state names found"

        # Check for duplicate step IDs within the flow
        step_ids: set[str] = set()
        for state in self.states:
            for step in state.steps:
                if step.id in step_ids:
                    return False, f"Duplicate step ID '{step.id}' found"
                step_ids.add(step.id)

        return True, None


class FlowExecutionStatus(str, Enum):
    """Status of flow execution."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_CONDITION = "waiting_condition"  # Waiting for a conditional transition
    COMPLETED = "completed"
    ABORTED = "aborted"
    ERROR = "error"


class FlowStatusResponse(BaseModel):
    """Response for flow status queries."""

    status: FlowExecutionStatus
    flow_id: Optional[str] = None
    current_state: Optional[str] = None
    current_step: Optional[str] = None
    error_message: Optional[str] = None
