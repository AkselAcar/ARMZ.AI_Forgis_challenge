"""Flow executor - state machine runner with error handling."""

import asyncio
import copy
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

from .schemas import (
    ErrorHandling,
    FlowExecutionStatus,
    FlowSchema,
    StepSchema,
)
from skills import get_skill, ExecutionContext, SkillResult

if TYPE_CHECKING:
    from executors import Executor

logger = logging.getLogger(__name__)

# Regex patterns for variable interpolation
VAR_PATTERN = re.compile(r"\{\{([\w.]+)\}\}")
LOOKUP_PATTERN = re.compile(r"\{\{lookup:([\w.]+):([\w.]+):([\w.]+)\}\}")
POP_PATTERN = re.compile(r"\{\{pop:([\w.]+):([\w.]+):([\w.]+)\}\}")
GET_PATTERN = re.compile(r"\{\{get:([\w.]+):([\w.]+):([\w.]+)\}\}")


class FlowPositionsExhausted(Exception):
    """Raised when a pop operation finds an empty list, signalling graceful flow completion."""

    pass


@dataclass
class StepExecutionResult:
    """Result of executing a single step."""

    step_id: str
    skill: str
    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    retries: int = 0
    duration_ms: float = 0


@dataclass
class FlowExecutionResult:
    """Result of executing a complete flow."""

    flow_id: str
    status: FlowExecutionStatus
    states_completed: list[str] = field(default_factory=list)
    steps_executed: list[StepExecutionResult] = field(default_factory=list)
    error_message: Optional[str] = None
    duration_ms: float = 0


# Event callback type for WebSocket broadcasting
EventCallback = Callable[[str, dict], None]


class FlowExecutor:
    """
    State machine executor for flows.

    Traverses states sequentially, executing steps within each state.
    Handles per-step error handling strategies.
    Emits events via callback for WebSocket telemetry.
    """

    def __init__(
        self,
        flow: FlowSchema,
        executors: dict[str, "Executor"],
        on_event: Optional[EventCallback] = None,
    ):
        self._flow = flow
        self._executors = executors
        self._on_event = on_event or (lambda t, d: None)
        self._abort_requested = False
        self._finish_requested = False
        self._paused = False
        self._pause_event = asyncio.Event()
        self._pause_event.set()  # Not paused initially
        self._status = FlowExecutionStatus.IDLE
        self._current_state: Optional[str] = None
        self._current_step: Optional[str] = None
        # Runtime variables: starts with flow-defined variables (lookup tables)
        self._variables: dict[str, Any] = copy.deepcopy(flow.variables)

    def request_abort(self) -> None:
        """Request abort of current execution."""
        logger.info(f"Abort requested for flow {self._flow.id}")
        self._abort_requested = True
        # Resume if paused to allow abort to proceed
        self._pause_event.set()

    def request_pause(self) -> None:
        """Request pause of current execution."""
        logger.info(f"Pause requested for flow {self._flow.id}")
        self._paused = True
        self._pause_event.clear()
        self._emit("flow_paused", {})

    def request_resume(self) -> None:
        """Request resume of paused execution."""
        logger.info(f"Resume requested for flow {self._flow.id}")
        self._paused = False
        self._pause_event.set()
        self._emit("flow_resumed", {})

    def request_finish(self) -> None:
        """Request graceful finish — complete current loop cycle then stop."""
        logger.info(f"Finish requested for flow {self._flow.id}")
        self._finish_requested = True

    def get_status(self) -> tuple[FlowExecutionStatus, Optional[str], Optional[str]]:
        """Get current execution status."""
        if self._paused:
            return FlowExecutionStatus.PAUSED, self._current_state, self._current_step
        return self._status, self._current_state, self._current_step

    def is_paused(self) -> bool:
        """Check if execution is currently paused."""
        return self._paused

    def _emit(self, event_type: str, data: dict) -> None:
        """Emit an event via callback."""
        self._on_event(event_type, {"flow_id": self._flow.id, **data})

    def _get_nested_value(self, path: str) -> Any:
        """
        Get a value from variables using dot notation.

        Examples:
            "label" -> self._variables["label"]
            "label_data.label" -> self._variables["label_data"]["label"]
        """
        parts = path.split(".")
        value = self._variables
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
            if value is None:
                return None
        return value

    def _resolve_value(self, value: Any) -> Any:
        """
        Resolve interpolation in a value.

        Supports:
        - {{var_name}} - get runtime variable
        - {{var.nested.path}} - get nested value from variable
        - {{lookup:map_name:key_path:default_key}} - lookup in map using variable as key
        - {{get:map_name:key_path:default_key}} - get first element from list without removing
        - {{pop:map_name:key_path:default_key}} - pop first element from list (mutating)
        """
        if isinstance(value, str):
            # Check for pop pattern: {{pop:map_name:key_path:default_key}}
            # Like lookup, but pops the first element from the list (mutates _variables).
            pop_match = POP_PATTERN.fullmatch(value)
            if pop_match:
                map_name, key_path, default_key = pop_match.groups()
                lookup_map = self._get_nested_value(map_name)
                if not isinstance(lookup_map, dict):
                    lookup_map = {}
                key_value = self._get_nested_value(key_path)
                if key_value is None:
                    key_value = default_key
                positions_list = lookup_map.get(key_value, lookup_map.get(default_key))
                if not isinstance(positions_list, list) or len(positions_list) == 0:
                    logger.info(
                        f"Pop {map_name}[{key_value}]: list empty, signalling completion"
                    )
                    raise FlowPositionsExhausted(
                        f"No more positions in {map_name}[{key_value}]"
                    )
                resolved = positions_list.pop(0)
                logger.debug(
                    f"Pop {map_name}[{key_value}] -> {resolved} "
                    f"({len(positions_list)} remaining)"
                )
                return resolved

            # Check for get pattern: {{get:map_name:key_path:default_key}}
            # Like pop, but returns first element WITHOUT removing it (non-mutating).
            get_match = GET_PATTERN.fullmatch(value)
            if get_match:
                map_name, key_path, default_key = get_match.groups()
                lookup_map = self._get_nested_value(map_name)
                if not isinstance(lookup_map, dict):
                    lookup_map = {}
                key_value = self._get_nested_value(key_path)
                if key_value is None:
                    key_value = default_key
                positions_list = lookup_map.get(key_value, lookup_map.get(default_key))
                if not isinstance(positions_list, list) or len(positions_list) == 0:
                    logger.warning(f"Get {map_name}[{key_value}]: list empty or not found")
                    return None
                resolved = positions_list[0]  # Get first element without removing
                logger.debug(f"Get {map_name}[{key_value}] -> {resolved}")
                return resolved

            # Check for lookup pattern: {{lookup:map_name:key_path:default_key}}
            lookup_match = LOOKUP_PATTERN.fullmatch(value)
            if lookup_match:
                map_name, key_path, default_key = lookup_match.groups()
                lookup_map = self._get_nested_value(map_name)
                if not isinstance(lookup_map, dict):
                    lookup_map = {}
                key_value = self._get_nested_value(key_path)
                if key_value is None:
                    key_value = default_key
                resolved = lookup_map.get(key_value, lookup_map.get(default_key))
                logger.debug(f"Lookup {map_name}[{key_value}] -> {resolved}")
                return resolved

            # Check for simple variable: {{var_name}} or {{var.nested.path}}
            var_match = VAR_PATTERN.fullmatch(value)
            if var_match:
                var_path = var_match.group(1)
                resolved = self._get_nested_value(var_path)
                logger.debug(f"Variable {var_path} -> {resolved}")
                return resolved

            return value

        elif isinstance(value, dict):
            return {k: self._resolve_value(v) for k, v in value.items()}

        elif isinstance(value, list):
            return [self._resolve_value(v) for v in value]

        return value

    def _resolve_params(self, params: dict) -> dict:
        """Resolve all interpolations in step parameters."""
        return self._resolve_value(params)

    async def _get_next_state_with_condition_polling(
        self, current_state: str, poll_interval_ms: int = 100
    ) -> Optional[str]:
        """
        Get next state, polling conditions if needed.

        If there are conditional transitions from current state:
        - If a sequential fallback also exists, evaluate conditions once
          and use the sequential transition as an "else" branch.
        - If no sequential fallback, poll conditions until one matches.
        - Respect pause/abort requests while polling.

        If no conditional transitions, fall back to sequential.

        Args:
            current_state: Current state name
            poll_interval_ms: Interval between condition checks

        Returns:
            Next state name or None
        """
        # Check if there are conditional transitions from this state
        if not self._flow.has_conditional_transitions(current_state):
            # No conditionals, use sequential transition
            return self._flow.get_sequential_transition(current_state)

        # Check for a sequential fallback (acts as "else" branch)
        sequential_fallback = self._flow.get_sequential_transition(current_state)

        if sequential_fallback is not None:
            # Mixed mode: evaluate conditions once, fall back to sequential
            next_state = self._flow.evaluate_conditional_transitions(
                current_state,
                variables=self._variables,
                executors=self._executors,
            )
            if next_state:
                logger.info(f"Condition met, transitioning to: {next_state}")
                return next_state
            logger.info(
                f"No condition met from '{current_state}', "
                f"using sequential fallback to '{sequential_fallback}'"
            )
            return sequential_fallback

        # No sequential fallback: poll conditional transitions until one matches
        self._emit("waiting_condition", {"state": current_state})
        logger.info(f"Waiting for condition from state: {current_state}")

        poll_count = 0
        while True:
            # Check for abort
            if self._abort_requested:
                return None

            # Check for pause
            await self._pause_event.wait()

            poll_count += 1

            # Evaluate conditional transitions
            next_state = self._flow.evaluate_conditional_transitions(
                current_state,
                variables=self._variables,
                executors=self._executors,
            )

            if next_state:
                logger.info(f"Condition met after {poll_count} polls, transitioning to: {next_state}")
                return next_state

            # Log every 50 polls (5 seconds at 100ms interval)
            if poll_count % 50 == 0:
                logger.info(f"Still waiting for condition from '{current_state}' (poll #{poll_count})")

            # Wait before next poll
            await asyncio.sleep(poll_interval_ms / 1000.0)

    async def execute(self) -> FlowExecutionResult:
        """
        Execute the flow from initial state.

        Returns:
            FlowExecutionResult with status and step results.
        """
        self._abort_requested = False
        self._status = FlowExecutionStatus.RUNNING
        start_time = time.monotonic()

        result = FlowExecutionResult(
            flow_id=self._flow.id, status=FlowExecutionStatus.RUNNING
        )

        self._emit("flow_started", {"name": self._flow.name})
        logger.info(f"Starting flow: {self._flow.id} ({self._flow.name})")

        current_state_name = self._flow.initial_state

        try:
            while current_state_name is not None:
                if self._abort_requested:
                    logger.info("Flow aborted by user")
                    result.status = FlowExecutionStatus.ABORTED
                    self._emit("flow_aborted", {})
                    break

                state = self._flow.get_state(current_state_name)
                if state is None:
                    error_msg = f"State '{current_state_name}' not found"
                    logger.error(error_msg)
                    result.status = FlowExecutionStatus.ERROR
                    result.error_message = error_msg
                    self._emit("flow_error", {"error": error_msg})
                    break

                self._current_state = current_state_name
                self._emit("state_entered", {"state": current_state_name})
                logger.info(f"Entering state: {current_state_name}")

                # Execute all steps in this state
                try:
                    for step in state.steps:
                        # Check for pause before each step
                        await self._pause_event.wait()

                        if self._abort_requested:
                            break

                        step_result = await self._execute_step(step)
                        result.steps_executed.append(step_result)

                        if not step_result.success:
                            # Error handling is done in _execute_step
                            # If we get here with failure, it means we should stop
                            if step.error_handling.strategy == ErrorHandling.STOP:
                                result.status = FlowExecutionStatus.ERROR
                                result.error_message = step_result.error
                                self._emit(
                                    "flow_error",
                                    {"error": step_result.error, "step_id": step.id},
                                )
                                return self._finalize_result(result, start_time)
                except FlowPositionsExhausted as e:
                    logger.info(f"Flow completing: {e}")
                    result.status = FlowExecutionStatus.COMPLETED
                    self._emit("flow_completed", {"status": "completed", "reason": str(e)})
                    return self._finalize_result(result, start_time)

                if self._abort_requested:
                    continue

                result.states_completed.append(current_state_name)
                self._emit("state_completed", {"state": current_state_name})

                # Get next state with condition polling
                current_state_name = await self._get_next_state_with_condition_polling(
                    current_state_name
                )
                if current_state_name is None and self._flow.loop and not self._finish_requested:
                    current_state_name = self._flow.initial_state
                    logger.info(f"Looping back to initial state: {current_state_name}")
                    self._emit("loop_restart", {"initial_state": current_state_name})
                elif current_state_name is None and self._finish_requested:
                    logger.info("Finish requested — exiting loop")

            # Flow completed successfully (if not aborted/errored)
            if result.status == FlowExecutionStatus.RUNNING:
                result.status = FlowExecutionStatus.COMPLETED
                self._emit("flow_completed", {"status": "completed"})
                logger.info(f"Flow completed: {self._flow.id}")

        except Exception as e:
            logger.exception(f"Unexpected error in flow execution: {e}")
            result.status = FlowExecutionStatus.ERROR
            result.error_message = str(e)
            self._emit("flow_error", {"error": str(e)})

        return self._finalize_result(result, start_time)

    def _finalize_result(
        self, result: FlowExecutionResult, start_time: float
    ) -> FlowExecutionResult:
        """Finalize result with timing and status update."""
        result.duration_ms = (time.monotonic() - start_time) * 1000
        self._status = result.status
        self._current_state = None
        self._current_step = None
        return result

    async def _execute_step(self, step: StepSchema) -> StepExecutionResult:
        """
        Execute a single step with error handling.

        Implements retry, skip, and fallback strategies.
        """
        self._current_step = step.id
        start_time = time.monotonic()

        self._emit(
            "step_started",
            {"step_id": step.id, "skill": step.skill, "executor": step.executor},
        )
        logger.info(f"Executing step: {step.id} (skill={step.skill})")

        result = StepExecutionResult(
            step_id=step.id,
            skill=step.skill,
            success=False,
        )

        # Get the skill
        try:
            skill = get_skill(step.skill)
        except KeyError as e:
            result.error = str(e)
            self._emit("step_error", {"step_id": step.id, "error": result.error})
            return result

        # Resolve parameter interpolation and parse
        try:
            resolved_params = self._resolve_params(step.params)
            logger.debug(f"Resolved params for {step.id}: {resolved_params}")
            params = skill.parse_params(resolved_params)
            valid, validation_error = await skill.validate(params)
            if not valid:
                result.error = f"Validation failed: {validation_error}"
                self._emit("step_error", {"step_id": step.id, "error": result.error})
                return result
        except Exception as e:
            result.error = f"Parameter error: {e}"
            self._emit("step_error", {"step_id": step.id, "error": result.error})
            return result

        # Create execution context with variables
        context = ExecutionContext(
            flow_id=self._flow.id,
            step_id=step.id,
            state_name=self._current_state or "",
            executor_type=step.executor,
            executors=self._executors,
            variables=self._variables,
        )

        # Execute with retry logic
        max_attempts = (
            step.error_handling.max_retries + 1
            if step.error_handling.strategy == ErrorHandling.RETRY
            else 1
        )

        last_error: Optional[str] = None

        for attempt in range(max_attempts):
            if self._abort_requested:
                result.error = "Aborted"
                return result

            try:
                # Execute with timeout
                skill_result = await asyncio.wait_for(
                    skill.execute(params, context),
                    timeout=step.timeout_ms / 1000.0,
                )

                if skill_result.success:
                    result.success = True
                    result.data = skill_result.data
                    result.retries = attempt
                    result.duration_ms = (time.monotonic() - start_time) * 1000

                    # Store result in variables if store_result is specified
                    if step.store_result:
                        self._variables[step.store_result] = skill_result.data
                        logger.info(f"Stored result in variable '{step.store_result}': {skill_result.data}")

                    self._emit(
                        "step_completed",
                        {
                            "step_id": step.id,
                            "result": skill_result.data,
                            "retries": attempt,
                        },
                    )
                    logger.info(f"Step completed: {step.id}")
                    return result
                else:
                    last_error = skill_result.error
                    logger.warning(
                        f"Step {step.id} failed (attempt {attempt + 1}): {last_error}"
                    )

            except asyncio.TimeoutError:
                last_error = f"Timeout after {step.timeout_ms}ms"
                logger.warning(f"Step {step.id} timed out (attempt {attempt + 1})")

            except Exception as e:
                last_error = str(e)
                logger.exception(f"Step {step.id} exception (attempt {attempt + 1})")

            # Retry delay (if not last attempt)
            if attempt < max_attempts - 1:
                self._emit(
                    "step_retry",
                    {
                        "step_id": step.id,
                        "attempt": attempt + 1,
                        "max_retries": step.error_handling.max_retries,
                    },
                )
                await asyncio.sleep(step.error_handling.retry_delay_ms / 1000.0)

        # All attempts failed
        result.error = last_error
        result.retries = max_attempts - 1
        result.duration_ms = (time.monotonic() - start_time) * 1000

        # Apply error handling strategy
        strategy = step.error_handling.strategy

        if strategy == ErrorHandling.SKIP:
            logger.warning(f"Step {step.id} failed but SKIP strategy - continuing")
            result.success = True  # Mark as "successful" for flow continuation
            self._emit(
                "step_skipped",
                {"step_id": step.id, "error": result.error, "strategy": "skip"},
            )
        elif strategy == ErrorHandling.FALLBACK:
            # TODO: Execute fallback skill
            logger.warning(f"FALLBACK strategy not yet implemented for step {step.id}")
            self._emit(
                "step_error",
                {"step_id": step.id, "error": result.error, "strategy": "fallback"},
            )
        else:
            self._emit(
                "step_error",
                {"step_id": step.id, "error": result.error, "strategy": "stop"},
            )

        return result
