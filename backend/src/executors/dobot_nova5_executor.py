"""Executor wrapping DobotNova5Node for motion control."""

import asyncio
import logging
import math
from typing import TYPE_CHECKING, Optional

from .base import Executor

if TYPE_CHECKING:
    from nodes.dobot_nova5_node import DobotNova5Node

logger = logging.getLogger(__name__)


class DobotNova5Executor(Executor):
    """
    Executor for DOBOT Nova 5 motion commands.

    Wraps DobotNova5Node and provides the same async interface as RobotExecutor
    so all existing skills (move_joint, move_linear, palletize) work unchanged.

    Key difference from UR: DOBOT uses ROS 2 service calls for motion. These
    services accept the command and return quickly (fire-and-forget). Motion
    completion is detected by polling joint stability — identical to how
    robot_executor handles move_linear.

    Service calls are blocking (use time.sleep internally) and must NOT be
    called directly from the asyncio thread. They are always dispatched via
    run_in_executor() so the event loop stays free during motion.
    """

    executor_type = "robot"

    def __init__(self, node: "DobotNova5Node"):
        self._node = node
        self._motion_poll_interval = 0.1  # seconds

    async def initialize(self) -> None:
        """Wait for DOBOT connection and enable the arm."""
        logger.info("DobotNova5Executor initializing...")
        timeout = 10.0
        elapsed = 0.0
        while self._node.get_joint_positions() is None and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1

        if self._node.get_joint_positions() is None:
            logger.warning("DOBOT joint positions not available after timeout — robot may not be connected")
            return

        # Enable robot (blocking call, dispatched to thread pool)
        loop = asyncio.get_event_loop()
        enabled = await loop.run_in_executor(None, self._node.enable_robot)
        if enabled:
            logger.info("DobotNova5Executor ready")
        else:
            logger.warning("DobotNova5Executor: EnableRobot failed — check DOBOT connection")

    async def shutdown(self) -> None:
        pass

    def is_ready(self) -> bool:
        return self._node.get_joint_positions() is not None

    async def move_joint(
        self,
        target_rad: list[float],
        acceleration: float = 1.4,
        velocity: float = 1.05,
        tolerance_rad: float = 0.02,
        timeout: float = 60.0,
    ) -> bool:
        """
        Execute a joint-space move and wait for completion.

        Args:
            target_rad: Target joint positions in radians (converted from skill params).
            acceleration: Unused for DOBOT (controlled by firmware speed settings).
            velocity: Unused for DOBOT (controlled by firmware speed settings).
            tolerance_rad: Position tolerance in radians for completion detection.
            timeout: Maximum time to wait for motion completion.

        Returns:
            True if motion completed within tolerance, False on timeout.
        """
        target_deg = [math.degrees(r) for r in target_rad]
        tolerance_deg = math.degrees(tolerance_rad)

        logger.info(f"DobotNova5Executor: Starting JointMovJ to {target_deg}")

        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            None, self._node.send_joint_move, *target_deg
        )
        if not ok:
            logger.error("DobotNova5Executor: JointMovJ service call failed")
            return False

        # Poll until target reached or timeout
        elapsed = 0.0
        while elapsed < timeout:
            if self._node.joints_at_target(target_deg, tolerance_deg=tolerance_deg):
                logger.info("DobotNova5Executor: Joint target reached")
                return True
            await asyncio.sleep(self._motion_poll_interval)
            elapsed += self._motion_poll_interval

        logger.error(f"DobotNova5Executor: Joint move timeout after {timeout}s")
        return False

    async def move_linear(
        self,
        pose: list[float],
        acceleration: float = 1.2,
        velocity: float = 0.25,
        timeout: float = 60.0,
    ) -> bool:
        """
        Execute a Cartesian linear move and wait for completion.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians (SI units).
            acceleration: Unused for DOBOT.
            velocity: Unused for DOBOT.
            timeout: Maximum time to wait for motion completion.

        Returns:
            True if motion completed (joints stable), False on timeout.
        """
        x, y, z, rx, ry, rz = pose
        # DOBOT firmware expects mm for xyz, degrees for rotations
        x_mm = x * 1000.0
        y_mm = y * 1000.0
        z_mm = z * 1000.0
        rx_deg = math.degrees(rx)
        ry_deg = math.degrees(ry)
        rz_deg = math.degrees(rz)

        logger.info(f"DobotNova5Executor: Starting MovL to [{x_mm:.1f},{y_mm:.1f},{z_mm:.1f}] mm")

        loop = asyncio.get_event_loop()
        ok = await loop.run_in_executor(
            None, self._node.send_linear_move, x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg
        )
        if not ok:
            logger.error("DobotNova5Executor: MovL service call failed")
            return False

        # Detect completion by waiting for joints to stop moving
        elapsed = 0.0
        prev_joints = None
        stable_count = 0
        stable_threshold = 3

        while elapsed < timeout:
            await asyncio.sleep(self._motion_poll_interval)
            elapsed += self._motion_poll_interval

            current = self._node.get_joint_positions()
            if current is None:
                continue

            if prev_joints is not None:
                max_diff = max(abs(c - p) for c, p in zip(current, prev_joints))
                if max_diff < 0.001:
                    stable_count += 1
                    if stable_count >= stable_threshold:
                        logger.info(f"DobotNova5Executor: Linear move complete (stable for {stable_count} polls)")
                        return True
                else:
                    stable_count = 0

            prev_joints = current

        logger.error(f"DobotNova5Executor: Linear move timeout after {timeout}s")
        return False

    async def set_tool_output(self, index: int, status: int) -> bool:
        """Control a tool digital output (ToolDO).

        Dual-solenoid pneumatic gripper:
          index=1, status=1 → close
          index=2, status=1 → open
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._node.tool_do, index, status)

    def get_joint_positions_deg(self) -> Optional[list[float]]:
        return self._node.get_joint_positions_deg()

    def get_state_summary(self) -> dict:
        return self._node.get_state_summary()
