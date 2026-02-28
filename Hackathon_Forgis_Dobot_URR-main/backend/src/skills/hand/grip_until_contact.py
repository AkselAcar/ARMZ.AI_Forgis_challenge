"""GripUntilContact skill — close fingers until stall-based contact is detected."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

FingerName = Literal["thumb", "index", "middle", "little"]


class GripUntilContactParams(BaseModel):
    """Parameters for the grip_until_contact skill."""

    speed: int = Field(
        default=25,
        ge=15,
        le=100,
        description="Closing speed (lower = gentler contact, recommended: 20-30)",
    )
    fingers: list[FingerName] = Field(
        default=["thumb", "index", "middle", "little"],
        description="Which fingers to close (ring/rotate excluded — no stall detection)",
    )
    min_contacts: int = Field(
        default=2,
        ge=1,
        le=4,
        description="Minimum number of fingers that must stall to count as a successful grip",
    )
    timeout_s: float = Field(
        default=5.0,
        ge=0.1,
        le=30.0,
        description="Max time to wait for contact before releasing fingers",
    )


@register_skill
class GripUntilContactSkill(Skill[GripUntilContactParams]):
    """Close fingers until stall-based contact is detected."""

    name = "grip_until_contact"
    executor_type = "hand"
    description = (
        "Close COVVI hand fingers at low speed until stall-based contact is detected. "
        "Holds grip on contact; releases and returns contacted=false on timeout."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return GripUntilContactParams

    async def validate(
        self, params: GripUntilContactParams
    ) -> tuple[bool, Optional[str]]:
        if not params.fingers:
            return False, "At least one finger must be specified"
        if params.min_contacts > len(params.fingers):
            return (
                False,
                f"min_contacts ({params.min_contacts}) cannot exceed "
                f"number of fingers ({len(params.fingers)})",
            )
        return True, None

    async def execute(
        self, params: GripUntilContactParams, context: ExecutionContext
    ) -> SkillResult:
        hand_executor = context.get_executor()

        if not hand_executor.is_ready():
            return SkillResult.fail("COVVI hand not connected")

        try:
            result = await hand_executor.grip_until_contact(
                speed=params.speed,
                fingers=params.fingers,
                min_contacts=params.min_contacts,
                timeout_s=params.timeout_s,
            )
            return SkillResult.ok(result)
        except Exception as e:
            return SkillResult.fail(f"grip_until_contact failed: {e}")
