"""ROS 2 node wrapping the COVVI prosthetic hand via the eci SDK."""

import logging
import os
import threading
import time
from typing import Optional

from rclpy.node import Node

logger = logging.getLogger(__name__)


class CovviHandNode(Node):
    """
    ROS 2 node that manages a persistent connection to the COVVI hand.

    The eci.CovviInterface uses a context manager for connection lifetime,
    so the connection is kept alive in a background thread (same pattern as
    CameraBridgeNode). Realtime finger position callbacks arrive at ~10Hz.

    Env vars:
        HAND_IP: IP address of the COVVI hand (default: 192.168.163.169)
    """

    def __init__(self):
        super().__init__("covvi_hand_node")

        self._hand_ip = os.environ.get("HAND_IP", "192.168.163.169")
        self._connected = False
        self._running = True
        self._hand = None  # CovviInterface instance (set inside background thread)
        self._hand_state: Optional[dict] = None
        self._hand_status: Optional[dict] = None
        self._lock = threading.Lock()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"CovviHandNode initialized — connecting to {self._hand_ip}"
        )

    def _run(self) -> None:
        """Open CovviInterface and keep the connection alive."""
        try:
            from eci import CovviInterface

            with CovviInterface(self._hand_ip) as hand:
                hand.setHandPowerOn()
                hand.callbackDigitPosnAll = self._on_hand_state
                hand.callbackDigitStatusAll = self._on_hand_status
                hand.setRealtimeCfg(digit_posn=True, digit_status=True)

                with self._lock:
                    self._hand = hand
                    self._connected = True

                self.get_logger().info(
                    f"Connected to COVVI hand at {self._hand_ip}"
                )

                # Keep the context manager alive until node is destroyed
                while self._running:
                    time.sleep(0.1)

                hand.disableAllRealtimeCfg()

        except ImportError:
            self.get_logger().error(
                "eci package not installed — COVVI hand unavailable"
            )
        except ConnectionRefusedError:
            self.get_logger().warning(
                f"Could not reach COVVI hand at {self._hand_ip}"
            )
        except Exception as e:
            self.get_logger().error(f"COVVI hand error: {e}")
        finally:
            with self._lock:
                self._connected = False
                self._hand = None

    def _on_hand_state(self, msg) -> None:
        """Realtime callback (~10Hz) for finger position updates."""
        with self._lock:
            self._hand_state = {
                "thumb": msg.thumb_pos,
                "index": msg.index_pos,
                "middle": msg.middle_pos,
                "ring": msg.ring_pos,
                "little": msg.little_pos,
                "rotate": msg.rotate_pos,
            }

    def _on_hand_status(self, msg) -> None:
        """Realtime callback (~10Hz) for stall/gripping status."""
        with self._lock:
            self._hand_status = {
                "thumb":  msg.thumb_stall  or msg.thumb_gripping,
                "index":  msg.index_stall  or msg.index_gripping,
                "middle": msg.middle_stall or msg.middle_gripping,
                "little": msg.little_stall or msg.little_gripping,
            }

    # ── Public interface (called from HandExecutor) ─────────────────────────

    def set_grip(self, grip_name: str) -> None:
        """Set a predefined grip by name (e.g. 'POWER', 'RELAXED')."""
        from eci.enums import CurrentGripID

        grip_id = getattr(CurrentGripID, grip_name, None)
        if grip_id is None:
            raise ValueError(f"Unknown grip name: {grip_name!r}")

        with self._lock:
            if self._hand is None:
                raise RuntimeError("COVVI hand not connected")
            self._hand.setCurrentGrip(grip_id)

    def set_finger_positions(self, speed: int = 50, **fingers) -> None:
        """
        Set individual finger positions.

        Args:
            speed: Movement speed (15-100).
            **fingers: Keyword args for each finger (thumb, index, middle,
                       ring, little, rotate), each 0-100. Omitted fingers
                       stay at their current position.
        """
        from eci.primitives.speed import Speed

        with self._lock:
            if self._hand is None:
                raise RuntimeError("COVVI hand not connected")
            self._hand.setDigitPosn(Speed(speed), **fingers)

    def get_hand_state(self) -> Optional[dict]:
        """Return the latest finger positions, or None if not yet received."""
        with self._lock:
            return dict(self._hand_state) if self._hand_state else None

    def get_hand_status(self) -> Optional[dict]:
        """Return latest stall/gripping flags per finger, or None if not yet received."""
        with self._lock:
            return dict(self._hand_status) if self._hand_status else None

    def stop_fingers(self) -> None:
        """Freeze all finger motors in place."""
        with self._lock:
            if self._hand is None:
                raise RuntimeError("COVVI hand not connected")
            self._hand.setDigitPosnStop()

    def is_connected(self) -> bool:
        """Return True if the hand is connected and operational."""
        return self._connected

    def destroy_node(self) -> None:
        self._running = False
        super().destroy_node()
