"""Palletize skill — move_linear with automatic position popping from a list."""

import logging
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

logger = logging.getLogger(__name__)


class PalletizeParams(BaseModel):
    """Parameters for the palletize skill."""

    positions_var: str = Field(
        ...,
        description="Name of the flow variable holding the positions map (e.g. 'place_positions')",
    )
    key: str = Field(
        ...,
        description="Zone key to look up in the positions map (e.g. 'product_A')",
    )
    default_key: str = Field(
        default="default",
        description="Fallback key if the zone key is not found in the map",
    )
    z_offset: float = Field(
        default=0.0,
        description="Vertical offset added to Z component before execution (meters). Positive = up.",
    )
    acceleration: float = Field(
        default=1.2,
        ge=0.01,
        le=3.0,
        description="Tool acceleration in m/s²",
    )
    velocity: float = Field(
        default=0.8,
        ge=0.01,
        le=1.0,
        description="Tool velocity in m/s",
    )


@register_skill
class PalletizeSkill(Skill[PalletizeParams]):
    """Move robot to the next position from a palletizing list, popping it after use."""

    name = "palletize"
    executor_type = "robot"
    description = "Move to next pallet position from a list, removing it for the next cycle"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return PalletizeParams

    async def validate(self, params: PalletizeParams) -> tuple[bool, Optional[str]]:
        return True, None

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize a zone key: strip, lowercase, collapse whitespace to underscore."""
        return "_".join(key.strip().lower().split())

    def _find_key(self, positions_map: dict, raw_key: str, default_key: str) -> str:
        """Find the best matching key in positions_map, with fuzzy normalization."""
        # Exact match first
        if raw_key in positions_map:
            return raw_key

        # Normalized match: compare normalized forms of all keys
        norm = self._normalize_key(raw_key)
        for map_key in positions_map:
            if map_key == default_key:
                continue
            if self._normalize_key(map_key) == norm:
                logger.info(f"Palletize: fuzzy matched key '{raw_key}' -> '{map_key}'")
                return map_key

        # Nothing matched — fall back to default
        logger.warning(
            f"Palletize: key '{raw_key}' (norm='{norm}') not found in "
            f"{list(positions_map.keys())}, using default '{default_key}'"
        )
        return default_key

    async def execute(
        self, params: PalletizeParams, context: ExecutionContext
    ) -> SkillResult:
        # Look up the positions map from flow variables
        positions_map = context.get_variable(params.positions_var)
        if not isinstance(positions_map, dict):
            return SkillResult.fail(
                f"Variable '{params.positions_var}' is not a dict",
                {"positions_var": params.positions_var},
            )

        # Find the best matching key (handles GPT label variations)
        matched_key = self._find_key(positions_map, params.key, params.default_key)
        positions_list = positions_map.get(matched_key)
        if not isinstance(positions_list, list) or len(positions_list) == 0:
            return SkillResult.fail(
                "positions_exhausted",
                {"positions_var": params.positions_var, "key": params.key,
                 "matched_key": matched_key, "remaining": 0},
            )

        # Pop the first position (mutates the list in-place)
        original_pose = list(positions_list.pop(0))
        remaining = len(positions_list)
        logger.info(
            f"Palletize: key='{params.key}' matched='{matched_key}' "
            f"-> {original_pose} ({remaining} remaining)"
        )

        # Apply z_offset to Z component for the actual move
        move_pose = list(original_pose)
        move_pose[2] += params.z_offset

        # Execute move_linear
        robot_executor = context.get_executor("robot")
        success = await robot_executor.move_linear(
            pose=move_pose,
            acceleration=params.acceleration,
            velocity=params.velocity,
        )

        if success:
            return SkillResult.ok({
                "target_pose": original_pose,
                "z_offset": params.z_offset,
                "remaining": remaining,
            })
        else:
            return SkillResult.fail(
                "Failed to reach pallet position",
                {"target_pose": original_pose, "z_offset": params.z_offset, "remaining": remaining},
            )
