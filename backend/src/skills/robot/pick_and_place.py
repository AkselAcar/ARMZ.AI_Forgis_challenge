"""Pick-and-place compound skill with Gemini vision zone routing.

Full cycle:
  1.  Move to pick pose
  2.  Vacuum ON
  3.  Read label via Gemini → determine zone
  4.  Pick next place pose from zone list (wraps around when exhausted)
  5.  Optional: move to waypoint (Zone_C detour)
  6.  Move to place pose
  7.  Vacuum OFF
  8.  Optional: Zone_C return waypoint (avoid collision)
"""

import asyncio
import logging
import math
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Read the product label in the image. Map it to one of these zones: "
    "Zone_A = MXP-30, Zone_B = MXP Speed, Zone_C = MXP Torque. "
    "Return ONLY the zone name (Zone_A, Zone_B, or Zone_C), nothing else. "
    "If you are unsure, return the most likely zone."
)


class PickAndPlaceParams(BaseModel):
    """Parameters for the pick_and_place skill."""

    pick_pose: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Pick target pose [x, y, z, rx, ry, rz] in metres and radians",
    )
    positions_var: str = Field(
        default="place_positions",
        description="Flow variable name containing a dict of zone -> list of poses",
    )
    default_zone: str = Field(
        default="default",
        description="Zone key to use when Gemini label is unrecognized",
    )
    waypoint_c_joints: Optional[list[float]] = Field(
        default=None,
        min_length=6,
        max_length=6,
        description="Joint angles (degrees) for Zone_C waypoint. None = no waypoint.",
    )
    waypoint_c_zone: str = Field(
        default="Zone_C",
        description="Zone name that triggers the waypoint_c_joints detour",
    )
    label_prompt: str = Field(
        default=_DEFAULT_PROMPT,
        description="Gemini prompt for zone classification",
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
        default=1.0,
        ge=0.01,
        le=2.0,
        description="Tool velocity for the pick descend/retract (m/s)",
    )
    place_velocity: float = Field(
        default=1.0,
        ge=0.01,
        le=2.0,
        description="Tool velocity for the place descend/retract (m/s)",
    )
    acceleration: float = Field(
        default=2.0,
        ge=0.01,
        le=5.0,
        description="Tool acceleration (m/s²)",
    )


@register_skill
class PickAndPlaceSkill(Skill[PickAndPlaceParams]):
    """Full pick-and-place cycle with Gemini vision zone routing."""

    name = "pick_and_place"
    executor_type = "robot"
    description = (
        "Execute a complete pick-and-place cycle using the vacuum gripper with "
        "Gemini vision zone routing. Reads label after picking to determine "
        "placement zone from a configurable positions map."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return PickAndPlaceParams

    async def validate(self, params: PickAndPlaceParams) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _normalize_key(key: str) -> str:
        return "_".join(key.strip().lower().split())

    def _find_key(self, positions_map: dict, raw_key: str, default_key: str) -> str:
        if raw_key in positions_map:
            return raw_key
        norm = self._normalize_key(raw_key)
        for map_key in positions_map:
            if map_key == default_key:
                continue
            if self._normalize_key(map_key) == norm:
                logger.info(f"pick_and_place: fuzzy matched zone '{raw_key}' -> '{map_key}'")
                return map_key
        logger.warning(
            f"pick_and_place: zone '{raw_key}' not found in {list(positions_map.keys())}, "
            f"using default '{default_key}'"
        )
        return default_key

    async def execute(
        self, params: PickAndPlaceParams, context: ExecutionContext
    ) -> SkillResult:
        robot = context.get_executor("robot")
        io = context.get_executor("io_robot")
        camera = context.get_executor("camera")

        pick = list(params.pick_pose)

        try:
            # ── 1. Move to pick ───────────────────────────────────
            logger.info("pick_and_place: moving to pick pose")
            ok = await robot.move_linear(
                pose=pick,
                acceleration=params.acceleration,
                velocity=params.pick_velocity,
            )
            if not ok:
                return SkillResult.fail("Failed to reach pick pose")

            # ── 2. Vacuum ON ──────────────────────────────────────
            logger.info("pick_and_place: vacuum ON (pin %d)", params.vacuum_pin)
            await io.set_digital_output(params.vacuum_pin, True)
            await asyncio.sleep(params.vacuum_settle_s)

            # ── 3. Read label via Gemini ──────────────────────────
            logger.info("pick_and_place: reading label via Gemini")
            label_result = await camera.read_label(
                prompt=params.label_prompt, use_bbox=False
            )
            zone = (
                label_result.get("label", "").strip()
                if label_result.get("success")
                else ""
            )
            logger.info(f"pick_and_place: Gemini zone = '{zone}'")

            # ── 4. Pick place pose (wrap-around index) ────────────
            positions_map = context.get_variable(params.positions_var)
            if not isinstance(positions_map, dict):
                await io.set_digital_output(params.vacuum_pin, False)
                return SkillResult.fail(
                    f"Flow variable '{params.positions_var}' is missing or not a dict"
                )
            matched_key = self._find_key(positions_map, zone, params.default_zone)
            positions_list = positions_map.get(matched_key, [])
            if not positions_list:
                await io.set_digital_output(params.vacuum_pin, False)
                return SkillResult.fail(f"No positions defined for zone '{matched_key}'")

            # Use a per-zone index stored in flow variables so the list is
            # never depleted — wraps back to position 0 after the last one.
            idx_var = f"_pnp_idx_{params.positions_var}"
            indices: dict = context.get_variable(idx_var) or {}
            current_idx = indices.get(matched_key, 0) % len(positions_list)
            place = list(positions_list[current_idx])
            indices[matched_key] = current_idx + 1
            context.set_variable(idx_var, indices)
            logger.info(
                f"pick_and_place: placing at {place} "
                f"(zone='{matched_key}', slot {current_idx + 1}/{len(positions_list)}, "
                f"next={indices[matched_key] % len(positions_list)})"
            )

            # ── 5. Zone_C waypoint detour (optional) ─────────────
            if (
                params.waypoint_c_joints
                and self._normalize_key(matched_key)
                == self._normalize_key(params.waypoint_c_zone)
            ):
                logger.info("pick_and_place: moving to Zone_C waypoint")
                target_rad = [math.radians(d) for d in params.waypoint_c_joints]
                ok = await robot.move_joint(
                    target_rad=target_rad,
                    acceleration=params.acceleration,
                    velocity=1.05,
                )
                if not ok:
                    await io.set_digital_output(params.vacuum_pin, False)
                    return SkillResult.fail("Failed to reach Zone_C waypoint")

            # ── 6. Move to place ──────────────────────────────────
            logger.info("pick_and_place: moving to place pose")
            ok = await robot.move_linear(
                pose=place,
                acceleration=params.acceleration,
                velocity=params.place_velocity,
            )
            if not ok:
                await io.set_digital_output(params.vacuum_pin, False)
                return SkillResult.fail("Failed to reach place pose")

            # ── 7. Vacuum OFF ─────────────────────────────────────
            logger.info("pick_and_place: vacuum OFF (pin %d)", params.vacuum_pin)
            await io.set_digital_output(params.vacuum_pin, False)
            await asyncio.sleep(params.vacuum_settle_s)

            # ── 8. Zone_C return waypoint (avoid collision) ───────
            if (
                params.waypoint_c_joints
                and self._normalize_key(matched_key)
                == self._normalize_key(params.waypoint_c_zone)
            ):
                logger.info("pick_and_place: returning via Zone_C waypoint")
                target_rad = [math.radians(d) for d in params.waypoint_c_joints]
                ok = await robot.move_joint(
                    target_rad=target_rad,
                    acceleration=params.acceleration,
                    velocity=1.05,
                )
                if not ok:
                    return SkillResult.fail("Failed to reach Zone_C return waypoint")

            logger.info("pick_and_place: cycle complete (zone='%s')", matched_key)
            return SkillResult.ok({
                "pick_pose": params.pick_pose,
                "place_pose": place,
                "zone": matched_key,
            })

        except Exception as e:
            logger.exception("pick_and_place: unexpected error")
            try:
                await io.set_digital_output(params.vacuum_pin, False)
            except Exception:
                pass
            return SkillResult.fail(f"Pick and place failed: {e}")