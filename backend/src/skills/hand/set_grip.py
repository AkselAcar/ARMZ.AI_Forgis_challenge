"""SetGrip skill for selecting a predefined COVVI hand grip."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

GripName = Literal[
    "POWER",
    "TRIPOD",
    "TRIPOD_OPEN",
    "PREC_OPEN",
    "PREC_CLOSED",
    "TRIGGER",
    "KEY",
    "FINGER",
    "CYLINDER",
    "COLUMN",
    "RELAXED",
    "GLOVE",
    "TAP",
    "GRAB",
]


class SetGripParams(BaseModel):
    """Parameters for the set_grip skill."""

    grip: GripName = Field(
        ...,
        description="Predefined grip name to activate",
    )


@register_skill
class SetGripSkill(Skill[SetGripParams]):
    """Activate a predefined grip on the COVVI prosthetic hand."""

    name = "set_grip"
    executor_type = "hand"
    description = "Set the COVVI hand to a predefined grip position"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return SetGripParams

    async def validate(self, params: SetGripParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: SetGripParams, context: ExecutionContext
    ) -> SkillResult:
        hand_executor = context.get_executor()

        if not hand_executor.is_ready():
            return SkillResult.fail("COVVI hand not connected")

        try:
            await hand_executor.set_grip(params.grip)
            return SkillResult.ok({"grip": params.grip})
        except Exception as e:
            return SkillResult.fail(
                f"Failed to set grip: {e}",
                {"grip": params.grip},
            )
