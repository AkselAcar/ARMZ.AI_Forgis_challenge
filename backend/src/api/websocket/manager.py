"""WebSocket connection manager for broadcasting events."""

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from .events import EventType, create_event

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts events.

    Thread-safe for use with asyncio.
    """

    def __init__(self):
        self._connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self._connections)}")

        # Send welcome event
        await self._send_to_one(
            websocket, create_event(EventType.CONNECTED, {"message": "Connected to flow execution system"})
        )

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a WebSocket connection."""
        async with self._lock:
            self._connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self._connections)}")

    async def broadcast(self, event_type: str, data: dict[str, Any]) -> None:
        """
        Broadcast an event to all connected clients.

        Args:
            event_type: Event type string.
            data: Event data (will be merged with type and timestamp).
        """
        event = create_event(event_type, data)
        await self._broadcast_raw(event)

    async def broadcast_event(self, event: dict[str, Any]) -> None:
        """Broadcast a pre-formed event dict."""
        await self._broadcast_raw(event)

    async def _broadcast_raw(self, event: dict[str, Any]) -> None:
        """Broadcast raw event dict to all connections."""
        if not self._connections:
            return

        message = json.dumps(event)
        disconnected: list[WebSocket] = []

        async with self._lock:
            connections = list(self._connections)

        for ws in connections:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Remove failed connections
        if disconnected:
            async with self._lock:
                for ws in disconnected:
                    self._connections.discard(ws)

    async def _send_to_one(self, websocket: WebSocket, event: dict[str, Any]) -> None:
        """Send an event to a single connection."""
        try:
            await websocket.send_text(json.dumps(event))
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")

    def connection_count(self) -> int:
        """Return current number of connections."""
        return len(self._connections)

    def create_event_callback(self):
        """
        Create a callback function for FlowExecutor events.

        Returns a sync function that schedules async broadcast.
        """

        def callback(event_type: str, data: dict) -> None:
            # Schedule the broadcast in the event loop
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.broadcast(event_type, data))
            except RuntimeError:
                # No running loop - log and skip
                logger.debug(f"No event loop for broadcast: {event_type}")

        return callback
