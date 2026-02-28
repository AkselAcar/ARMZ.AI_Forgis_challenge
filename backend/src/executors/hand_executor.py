"""Executor for the COVVI prosthetic hand."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .base import Executor

if TYPE_CHECKING:
    from nodes.covvi_hand_node import CovviHandNode

logger = logging.getLogger(__name__)


class HandExecutor(Executor):
    """
    Executor for COVVI hand operations.

    Wraps CovviHandNode and provides an async interface for skills.
    """

    executor_type = "hand"

    def __init__(self, hand_node: "CovviHandNode"):
        self._node = hand_node

    async def initialize(self) -> None:
        """Wait for the hand connection to be established."""
        logger.info("HandExecutor initializing...")
        timeout = 10.0
        elapsed = 0.0
        while not self._node.is_connected() and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        if self._node.is_connected():
            logger.info("HandExecutor ready")
        else:
            logger.warning(
                "HandExecutor: COVVI hand not reachable after timeout — "
                "hand skills will fail until connection is established"
            )

    async def shutdown(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self._node.is_connected()

    async def set_grip(self, grip_name: str) -> None:
        """Set a predefined grip by name."""
        logger.info(f"HandExecutor: setting grip={grip_name!r}")
        self._node.set_grip(grip_name)

    async def set_finger_positions(self, speed: int = 50, **fingers) -> None:
        """Set individual finger positions."""
        logger.info(f"HandExecutor: set_finger_positions speed={speed} fingers={fingers}")
        self._node.set_finger_positions(speed, **fingers)
        await asyncio.sleep(0.05)  # Brief delay for propagation

    def get_hand_state(self) -> Optional[dict]:
        """Return the latest finger positions."""
        return self._node.get_hand_state()

    def get_hand_status(self) -> Optional[dict]:
        """Return latest stall/gripping flags per finger."""
        return self._node.get_hand_status()

    async def grip_until_contact(
        self,
        speed: int,
        fingers: list,
        min_contacts: int,
        timeout_s: float,
    ) -> dict:
        """
        Close fingers at the given speed until min_contacts stall, or release on timeout.

        Returns:
            {"contacted": bool, "contact_fingers": list[str]}
        """
        self._node.set_finger_positions(speed, **{f: 100 for f in fingers})

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            await asyncio.sleep(0.05)
            status = self._node.get_hand_status()
            if status:
                contacted = [f for f in fingers if status.get(f, False)]
                if len(contacted) >= min_contacts:
                    self._node.stop_fingers()
                    return {"contacted": True, "contact_fingers": contacted}

        # Timeout — open fingers back up
        self._node.set_finger_positions(50, **{f: 0 for f in fingers})
        return {"contacted": False, "contact_fingers": []}
