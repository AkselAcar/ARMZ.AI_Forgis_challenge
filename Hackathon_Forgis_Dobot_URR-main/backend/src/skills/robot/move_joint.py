"""MoveJoint skill for moving the robot to target joint positions."""

import math
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class MoveJointParams(BaseModel):
    """Parameters for the move_joint skill."""

    target_joints_deg: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Target joint positions in degrees [j0, j1, j2, j3, j4, j5]",
    )
    acceleration: float = Field(
        default=2.0,
        ge=0.1,
        le=2.0,
        description="Joint acceleration in rad/s²",
    )
    velocity: float = Field(
        default=2.0,
        ge=0.1,
        le=2.0,
        description="Joint velocity in rad/s",
    )
    tolerance_deg: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Position tolerance in degrees to consider motion complete",
    )


@register_skill
class MoveJointSkill(Skill[MoveJointParams]):
    """Move robot to target joint positions using movej command."""

    name = "move_joint"
    executor_type = "robot"
    description = "Move robot joints to specified positions in degrees"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return MoveJointParams

    async def validate(self, params: MoveJointParams) -> tuple[bool, Optional[str]]:
        # Validate joint limits (UR3 typical limits)
        joint_limits_deg = [
            (-360, 360),  # Base
            (-360, 360),  # Shoulder
            (-360, 360),  # Elbow
            (-360, 360),  # Wrist 1
            (-360, 360),  # Wrist 2
            (-360, 360),  # Wrist 3
        ]

        for i, (joint_deg, (min_deg, max_deg)) in enumerate(
            zip(params.target_joints_deg, joint_limits_deg)
        ):
            if not min_deg <= joint_deg <= max_deg:
                return False, f"Joint {i} ({joint_deg}°) outside limits [{min_deg}, {max_deg}]"

        return True, None

    async def execute(
        self, params: MoveJointParams, context: ExecutionContext
    ) -> SkillResult:
        robot_executor = context.get_executor("robot")

        # Convert degrees to radians
        target_rad = [math.radians(deg) for deg in params.target_joints_deg]

        # Execute the move
        success = await robot_executor.move_joint(
            target_rad=target_rad,
            acceleration=params.acceleration,
            velocity=params.velocity,
            tolerance_rad=math.radians(params.tolerance_deg),
        )

        if success:
            return SkillResult.ok(
                {"target_joints_deg": params.target_joints_deg, "reached": True}
            )
        else:
            return SkillResult.fail(
                "Failed to reach target position",
                {"target_joints_deg": params.target_joints_deg, "reached": False},
            )
