"""WebSocket components for real-time telemetry."""

from .events import EventType, create_event
from .manager import WebSocketManager

__all__ = ["EventType", "WebSocketManager", "create_event"]
