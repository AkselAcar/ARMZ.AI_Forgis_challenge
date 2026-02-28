"""Robot executor wrapping RobotNode for motion control."""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from .base import Executor

if TYPE_CHECKING:
    from nodes.ur_node import RobotNode

logger = logging.getLogger(__name__)


class RobotExecutor(Executor):
    """
    Executor for robot motion commands.

    Wraps RobotNode and provides async interface for skills.
    Handles the URScript control pattern: send command, poll completion, resend program.
    """

    executor_type = "robot"

    def __init__(self, robot_node: "RobotNode"):
        self._robot = robot_node
        self._motion_poll_interval = 0.1  # seconds

    async def initialize(self) -> None:
        """Wait for robot connection and set TCP."""
        logger.info("RobotExecutor initializing...")
        timeout = 10.0
        elapsed = 0.0
        while self._robot.get_joint_positions() is None and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1
        if self._robot.get_joint_positions() is None:
            logger.warning("Robot joint positions not available after timeout")
        else:
            # Set TCP offset once connected
            self._robot.set_tcp()
            logger.info("RobotExecutor ready (TCP configured)")

    async def shutdown(self) -> None:
        """No cleanup needed - RobotNode lifecycle managed elsewhere."""
        pass

    def is_ready(self) -> bool:
        """Check if robot is connected (has joint positions)."""
        return self._robot.get_joint_positions() is not None

    async def move_joint(
        self,
        target_rad: list[float],
        acceleration: float = 1.4,
        velocity: float = 1.05,
        tolerance_rad: float = 0.02,
        timeout: float = 60.0,
    ) -> bool:
        """
        Execute a movej command and wait for completion.

        Args:
            target_rad: Target joint positions in radians.
            acceleration: Joint acceleration in rad/s².
            velocity: Joint velocity in rad/s.
            tolerance_rad: Position tolerance in radians.
            timeout: Maximum time to wait for motion completion.

        Returns:
            True if motion completed successfully, False otherwise.
        """
        logger.info(f"RobotExecutor: Starting movej to {target_rad}")

        # Send the motion command (interrupts External Control)
        self._robot.send_movej(target_rad, accel=acceleration, vel=velocity)

        # Poll until target reached or timeout
        elapsed = 0.0
        while elapsed < timeout:
            if self._robot.joints_at_target(target_rad, tolerance=tolerance_rad):
                logger.info("RobotExecutor: Target reached")
                # Restore External Control
                await asyncio.sleep(0.2)  # Brief delay before resend
                self._robot.resend_robot_program()
                return True
            await asyncio.sleep(self._motion_poll_interval)
            elapsed += self._motion_poll_interval

        logger.error(f"RobotExecutor: Motion timeout after {timeout}s")
        self._robot.resend_robot_program()
        return False

    async def move_linear(
        self,
        pose: list[float],
        acceleration: float = 1.2,
        velocity: float = 0.25,
        timeout: float = 60.0,
    ) -> bool:
        """
        Execute a movel command and wait for completion.

        Args:
            pose: Target pose [x, y, z, rx, ry, rz] in meters and radians.
            acceleration: Tool acceleration in m/s².
            velocity: Tool velocity in m/s.
            timeout: Maximum time to wait for motion completion.

        Returns:
            True if motion completed successfully, False otherwise.
        """
        logger.info(f"RobotExecutor: Starting movel to {pose}")

        self._robot.send_movel(pose, accel=acceleration, vel=velocity)

        # Wait for motion to complete by detecting when joints stop moving
        elapsed = 0.0
        prev_joints = None
        stable_count = 0
        stable_threshold = 3  # Number of consecutive stable readings

        while elapsed < timeout:
            await asyncio.sleep(self._motion_poll_interval)
            elapsed += self._motion_poll_interval

            current_joints = self._robot.get_joint_positions()
            if current_joints is None:
                continue

            if prev_joints is not None:
                # Check if joints have stopped moving (all within tolerance)
                max_diff = max(abs(c - p) for c, p in zip(current_joints, prev_joints))
                if max_diff < 0.001:  # Less than ~0.06 degrees movement
                    stable_count += 1
                    if stable_count >= stable_threshold:
                        logger.info(f"RobotExecutor: Motion complete (stable for {stable_count} polls)")
                        await asyncio.sleep(0.2)  # Brief delay before resend
                        self._robot.resend_robot_program()
                        return True
                else:
                    stable_count = 0

            prev_joints = current_joints

        logger.error(f"RobotExecutor: Motion timeout after {timeout}s")
        self._robot.resend_robot_program()
        return False

    async def set_digital_output(self, pin: int, value: bool) -> None:
        """
        Set a digital output pin.

        Uses secondary script so it doesn't interrupt motion.
        """
        logger.info(f"RobotExecutor: Setting DO[{pin}] = {value}")
        self._robot.set_digital_output(pin, value)
        await asyncio.sleep(0.05)  # Brief delay for I/O propagation

    def get_joint_positions_deg(self) -> Optional[list[float]]:
        """Get current joint positions in degrees."""
        return self._robot.get_joint_positions_deg()

    def get_state_summary(self) -> dict:
        """Get robot state summary."""
        return self._robot.get_state_summary()
