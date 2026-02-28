"""I/O executor for digital input/output operations."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional, Any

from .base import Executor

if TYPE_CHECKING:
    from nodes.ur_node import RobotNode

logger = logging.getLogger(__name__)


class IOExecutor(Executor):
    """
    Executor for digital I/O operations.

    Works with any node that has _io_states, get_digital_input,
    get_digital_output, and set_digital_output.
    """

    executor_type = "io"

    def __init__(self, io_node: Any, executor_type: str = "io"):
        self._node = io_node
        self.executor_type = executor_type

    async def initialize(self) -> None:
        """Wait for I/O state availability."""
        logger.info(f"IOExecutor ({self.executor_type}) initializing...")
        timeout = 10.0
        elapsed = 0.0
        while self._node._io_states is None and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        if self._node._io_states is None:
            logger.warning(f"IOExecutor ({self.executor_type}): I/O states not available after timeout")
        else:
            logger.info(f"IOExecutor ({self.executor_type}) ready")

    async def shutdown(self) -> None:
        """No cleanup needed."""
        pass

    def is_ready(self) -> bool:
        """Check if I/O states are available."""
        return self._node._io_states is not None

    async def get_digital_input(self, pin: int) -> Optional[bool]:
        """
        Read a digital input pin.

        Args:
            pin: Digital input pin number.

        Returns:
            Current pin state, or None if unavailable.
        """
        return self._node.get_digital_input(pin)

    async def get_digital_output(self, pin: int) -> Optional[bool]:
        """
        Read a digital output pin state.

        Args:
            pin: Digital output pin number.

        Returns:
            Current pin state, or None if unavailable.
        """
        return self._node.get_digital_output(pin)

    async def set_digital_output(self, pin: int, value: bool) -> None:
        """
        Set a digital output pin.

        Args:
            pin: Digital output pin number.
            value: Desired pin state.
        """
        logger.info(f"IOExecutor ({self.executor_type}): Setting DO[{pin}] = {value}")
        self._node.set_digital_output(pin, value)
        await asyncio.sleep(0.05)  # Brief delay for I/O propagation

    async def wait_for_digital_input(
        self,
        pin: int,
        expected_value: bool,
        timeout: float = 30.0,
        poll_interval: float = 0.1,
    ) -> bool:
        """
        Wait for a digital input to reach expected value.

        Args:
            pin: Digital input pin number.
            expected_value: Value to wait for.
            timeout: Maximum wait time in seconds.
            poll_interval: Polling interval in seconds.

        Returns:
            True if expected value reached, False if timeout.
        """
        elapsed = 0.0
        while elapsed < timeout:
            current = await self.get_digital_input(pin)
            if current == expected_value:
                return True
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        return False
