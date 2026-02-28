"""Flow manager - orchestration layer for flow execution."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .executor import FlowExecutor, FlowExecutionResult
from .loader import FlowLoader
from .schemas import FlowExecutionStatus, FlowSchema, FlowStatusResponse

if TYPE_CHECKING:
    from api.websocket.manager import WebSocketManager
    from executors import Executor

logger = logging.getLogger(__name__)


class FlowManager:
    """
    Orchestration layer for flow execution.

    - Enforces single flow execution (rejects new requests while busy)
    - Manages flow loading/saving
    - Creates executors and FlowExecutor instances
    - Provides status queries
    """

    def __init__(
        self,
        executors: dict[str, "Executor"],
        ws_manager: "WebSocketManager",
        flows_dir: str = "/app/flows",
    ):
        self._executors = executors
        self._ws_manager = ws_manager
        self._loader = FlowLoader(flows_dir)

        self._current_executor: Optional[FlowExecutor] = None
        self._current_task: Optional[asyncio.Task] = None
        self._last_result: Optional[FlowExecutionResult] = None
        self._lock = asyncio.Lock()

    # --- Flow CRUD ---

    def list_flows(self) -> list[str]:
        """List all available flow IDs."""
        return self._loader.list_flows()

    def get_flow(self, flow_id: str) -> Optional[FlowSchema]:
        """Get a flow by ID."""
        return self._loader.load(flow_id)

    def save_flow(self, flow: FlowSchema) -> tuple[bool, Optional[str]]:
        """
        Save a flow.

        Returns:
            Tuple of (success, error_message).
        """
        # Validate flow structure
        valid, error = flow.validate_flow()
        if not valid:
            return False, error

        if self._loader.save(flow):
            return True, None
        return False, "Failed to save flow"

    def delete_flow(self, flow_id: str) -> bool:
        """Delete a flow."""
        return self._loader.delete(flow_id)

    # --- Execution Control ---

    async def start_flow(self, flow_id: str) -> tuple[bool, str]:
        """
        Start executing a flow.

        Returns:
            Tuple of (started, message).
        """
        async with self._lock:
            # Check if already running
            if self._is_running_locked():
                return False, "A flow is already running"

            # Load the flow
            flow = self._loader.load(flow_id)
            if flow is None:
                return False, f"Flow '{flow_id}' not found"

            # Validate
            valid, error = flow.validate_flow()
            if not valid:
                return False, f"Flow validation failed: {error}"

            # Create executor
            self._current_executor = FlowExecutor(
                flow=flow,
                executors=self._executors,
                on_event=self._ws_manager.create_event_callback(),
            )

            # Start execution task
            self._current_task = asyncio.create_task(self._run_flow())
            logger.info(f"Started flow: {flow_id}")
            return True, f"Flow '{flow_id}' started"

    async def _run_flow(self) -> None:
        """Execute the flow and store result."""
        if self._current_executor is None:
            return

        try:
            self._last_result = await self._current_executor.execute()
        except Exception as e:
            logger.exception(f"Flow execution error: {e}")
            self._last_result = FlowExecutionResult(
                flow_id=self._current_executor._flow.id,
                status=FlowExecutionStatus.ERROR,
                error_message=str(e),
            )
        finally:
            async with self._lock:
                self._current_executor = None
                self._current_task = None

    async def abort_flow(self) -> tuple[bool, str]:
        """
        Abort the currently running flow.

        Returns:
            Tuple of (aborted, message).
        """
        async with self._lock:
            if not self._is_running_locked():
                return False, "No flow is currently running"

            if self._current_executor:
                self._current_executor.request_abort()
                logger.info("Abort requested")
                return True, "Abort requested"

            return False, "No executor to abort"

    async def finish_flow(self) -> tuple[bool, str]:
        """
        Request graceful finish — complete current loop cycle then stop.

        Returns:
            Tuple of (success, message).
        """
        async with self._lock:
            if not self._is_running_locked():
                return False, "No flow is currently running"

            if self._current_executor:
                self._current_executor.request_finish()
                logger.info("Finish requested")
                return True, "Finish requested — flow will stop after current cycle"

            return False, "No executor to finish"

    async def pause_flow(self) -> tuple[bool, str]:
        """
        Pause the currently running flow.

        Returns:
            Tuple of (paused, message).
        """
        async with self._lock:
            if not self._is_running_locked():
                return False, "No flow is currently running"

            if self._current_executor:
                if self._current_executor.is_paused():
                    return False, "Flow is already paused"
                self._current_executor.request_pause()
                logger.info("Pause requested")
                return True, "Flow paused"

            return False, "No executor to pause"

    async def resume_flow(self) -> tuple[bool, str]:
        """
        Resume a paused flow.

        Returns:
            Tuple of (resumed, message).
        """
        async with self._lock:
            if not self._is_running_locked():
                return False, "No flow is currently running"

            if self._current_executor:
                if not self._current_executor.is_paused():
                    return False, "Flow is not paused"
                self._current_executor.request_resume()
                logger.info("Resume requested")
                return True, "Flow resumed"

            return False, "No executor to resume"

    def _is_running_locked(self) -> bool:
        """Check if a flow is running (must hold lock)."""
        return (
            self._current_task is not None
            and not self._current_task.done()
        )

    # --- Status ---

    def get_status(self) -> FlowStatusResponse:
        """Get current execution status."""
        if self._current_executor is not None and self._current_task is not None:
            if not self._current_task.done():
                status, state, step = self._current_executor.get_status()
                return FlowStatusResponse(
                    status=status,
                    flow_id=self._current_executor._flow.id,
                    current_state=state,
                    current_step=step,
                )

        # Not running - return last result status or idle
        if self._last_result is not None:
            return FlowStatusResponse(
                status=self._last_result.status,
                flow_id=self._last_result.flow_id,
                error_message=self._last_result.error_message,
            )

        return FlowStatusResponse(status=FlowExecutionStatus.IDLE)

    def get_last_result(self) -> Optional[FlowExecutionResult]:
        """Get the result of the last flow execution."""
        return self._last_result

    def is_running(self) -> bool:
        """Check if a flow is currently running."""
        return (
            self._current_task is not None
            and not self._current_task.done()
        )
