"""MoveLinear skill for moving the robot in a straight line to a Cartesian pose."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class MoveLinearParams(BaseModel):
    """Parameters for the move_linear skill."""

    target_pose: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Target pose [x, y, z, rx, ry, rz] in meters and radians",
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
class MoveLinearSkill(Skill[MoveLinearParams]):
    """Move robot TCP in a straight line to a Cartesian pose using movel command."""

    name = "move_linear"
    executor_type = "robot"
    description = "Move robot TCP linearly to a Cartesian pose [x, y, z, rx, ry, rz]"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return MoveLinearParams

    async def validate(self, params: MoveLinearParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: MoveLinearParams, context: ExecutionContext
    ) -> SkillResult:
        robot_executor = context.get_executor("robot")

        # Apply z_offset to Z component
        pose = list(params.target_pose)
        pose[2] += params.z_offset

        success = await robot_executor.move_linear(
            pose=pose,
            acceleration=params.acceleration,
            velocity=params.velocity,
        )

        if success:
            return SkillResult.ok({
                "target_pose": params.target_pose,
                "z_offset": params.z_offset,
            })
        else:
            return SkillResult.fail(
                "Failed to reach target pose",
                {"target_pose": params.target_pose, "z_offset": params.z_offset},
            )
