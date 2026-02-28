"""Pick-and-place compound skill for the UR3 + vacuum gripper setup.

Encapsulates the full cycle from the test_vision_speed flow:
  1. Descend to pick pose
  2. Activate vacuum
  3. Retract upward
  4. Move to place pose (approach above)
  5. Descend to place pose
  6. Release vacuum
  7. Retract upward
"""

import asyncio
import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

logger = logging.getLogger(__name__)


class PickAndPlaceParams(BaseModel):
    """Parameters for the pick_and_place skill."""

    pick_pose: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Pick target pose [x, y, z, rx, ry, rz] in metres and radians",
    )
    place_pose: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Place target pose [x, y, z, rx, ry, rz] in metres and radians",
    )
    approach_z_offset: float = Field(
        default=0.15,
        ge=0.01,
        le=0.5,
        description="Height above pick/place pose for approach & retract (metres)",
    )
    vacuum_pin: int = Field(
        default=0,
        ge=0,
        le=7,
        description="Digital output pin controlling the vacuum gripper",
    )
    vacuum_settle_s: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="Seconds to wait after vacuum on/off for pressure to settle",
    )
    pick_velocity: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Tool velocity for the pick descend/retract (m/s)",
    )
    place_velocity: float = Field(
        default=0.5,
        ge=0.01,
        le=1.0,
        description="Tool velocity for the place descend/retract (m/s)",
    )
    acceleration: float = Field(
        default=1.2,
        ge=0.01,
        le=3.0,
        description="Tool acceleration (m/s²)",
    )


@register_skill
class PickAndPlaceSkill(Skill[PickAndPlaceParams]):
    """Full pick-and-place cycle: descend → vacuum on → retract → move → descend → vacuum off → retract."""

    name = "pick_and_place"
    executor_type = "robot"
    description = (
        "Execute a complete pick-and-place cycle using the vacuum gripper. "
        "Descends to pick pose, activates vacuum, retracts, moves to place pose, "
        "releases vacuum, and retracts."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return PickAndPlaceParams

    async def validate(self, params: PickAndPlaceParams) -> tuple[bool, Optional[str]]:
        if params.approach_z_offset <= 0:
            return False, "approach_z_offset must be positive"
        return True, None

    async def execute(
        self, params: PickAndPlaceParams, context: ExecutionContext
    ) -> SkillResult:
        robot = context.get_executor("robot")
        io = context.get_executor("io_robot")

        pick = list(params.pick_pose)
        place = list(params.place_pose)
        z_off = params.approach_z_offset

        # Helper: apply z offset to a pose
        def with_z_offset(pose: list[float], offset: float) -> list[float]:
            p = list(pose)
            p[2] += offset
            return p

        try:
            # ── 1. Approach above pick ────────────────────────────
            logger.info("pick_and_place: approaching pick pose from above")
            ok = await robot.move_linear(
                pose=with_z_offset(pick, z_off),
                acceleration=params.acceleration,
                velocity=params.pick_velocity,
            )
            if not ok:
                return SkillResult.fail("Failed to reach pick approach pose")

            # ── 2. Descend to pick ────────────────────────────────
            logger.info("pick_and_place: descending to pick pose")
            ok = await robot.move_linear(
                pose=pick,
                acceleration=params.acceleration,
                velocity=params.pick_velocity * 0.5,  # slower final approach
            )
            if not ok:
                return SkillResult.fail("Failed to descend to pick pose")

            # ── 3. Vacuum ON ──────────────────────────────────────
            logger.info("pick_and_place: vacuum ON (pin %d)", params.vacuum_pin)
            await io.set_digital_output(params.vacuum_pin, True)
            await asyncio.sleep(params.vacuum_settle_s)

            # ── 4. Retract from pick ──────────────────────────────
            logger.info("pick_and_place: retracting from pick")
            ok = await robot.move_linear(
                pose=with_z_offset(pick, z_off),
                acceleration=params.acceleration,
                velocity=params.pick_velocity,
            )
            if not ok:
                return SkillResult.fail("Failed to retract from pick pose")

            # ── 5. Approach above place ───────────────────────────
            logger.info("pick_and_place: approaching place pose from above")
            ok = await robot.move_linear(
                pose=with_z_offset(place, z_off),
                acceleration=params.acceleration,
                velocity=params.place_velocity,
            )
            if not ok:
                return SkillResult.fail("Failed to reach place approach pose")

            # ── 6. Descend to place ───────────────────────────────
            logger.info("pick_and_place: descending to place pose")
            ok = await robot.move_linear(
                pose=place,
                acceleration=params.acceleration,
                velocity=params.place_velocity * 0.5,  # slower final
            )
            if not ok:
                return SkillResult.fail("Failed to descend to place pose")

            # ── 7. Vacuum OFF ─────────────────────────────────────
            logger.info("pick_and_place: vacuum OFF (pin %d)", params.vacuum_pin)
            await io.set_digital_output(params.vacuum_pin, False)
            await asyncio.sleep(params.vacuum_settle_s)

            # ── 8. Retract from place ─────────────────────────────
            logger.info("pick_and_place: retracting from place")
            ok = await robot.move_linear(
                pose=with_z_offset(place, z_off),
                acceleration=params.acceleration,
                velocity=params.place_velocity,
            )
            if not ok:
                return SkillResult.fail("Failed to retract from place pose")

            logger.info("pick_and_place: cycle complete")
            return SkillResult.ok({
                "pick_pose": params.pick_pose,
                "place_pose": params.place_pose,
            })

        except Exception as e:
            logger.exception("pick_and_place: unexpected error")
            # Safety: turn off vacuum on error
            try:
                await io.set_digital_output(params.vacuum_pin, False)
            except Exception:
                pass
            return SkillResult.fail(f"Pick and place failed: {e}")
