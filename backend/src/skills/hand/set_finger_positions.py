"""SetFingerPositions skill for individual finger control on the COVVI hand."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class SetFingerPositionsParams(BaseModel):
    """Parameters for the set_finger_positions skill."""

    speed: int = Field(
        default=50,
        ge=15,
        le=100,
        description="Movement speed (15=minimum firmware speed, 100=maximum)",
    )
    thumb: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Thumb position (0=open, 100=closed). Omit to leave unchanged.",
    )
    index: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Index finger position (0=open, 100=closed). Omit to leave unchanged.",
    )
    middle: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Middle finger position (0=open, 100=closed). Omit to leave unchanged.",
    )
    ring: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Ring finger position (0=open, 100=closed). Omit to leave unchanged.",
    )
    little: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Little finger position (0=open, 100=closed). Omit to leave unchanged.",
    )
    rotate: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Wrist rotation position (0=open, 100=closed). Omit to leave unchanged.",
    )


@register_skill
class SetFingerPositionsSkill(Skill[SetFingerPositionsParams]):
    """Set individual finger positions on the COVVI prosthetic hand."""

    name = "set_finger_positions"
    executor_type = "hand"
    description = (
        "Set individual COVVI hand finger positions (0=open, 100=closed). "
        "Only specified fingers move; others stay in place."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return SetFingerPositionsParams

    async def validate(
        self, params: SetFingerPositionsParams
    ) -> tuple[bool, Optional[str]]:
        fingers = {
            k: v
            for k, v in {
                "thumb": params.thumb,
                "index": params.index,
                "middle": params.middle,
                "ring": params.ring,
                "little": params.little,
                "rotate": params.rotate,
            }.items()
            if v is not None
        }
        if not fingers:
            return False, "At least one finger position must be specified"
        return True, None

    async def execute(
        self, params: SetFingerPositionsParams, context: ExecutionContext
    ) -> SkillResult:
        hand_executor = context.get_executor()

        if not hand_executor.is_ready():
            return SkillResult.fail("COVVI hand not connected")

        # Only pass fingers that were explicitly set
        fingers = {
            k: v
            for k, v in {
                "thumb": params.thumb,
                "index": params.index,
                "middle": params.middle,
                "ring": params.ring,
                "little": params.little,
                "rotate": params.rotate,
            }.items()
            if v is not None
        }

        try:
            await hand_executor.set_finger_positions(params.speed, **fingers)
            return SkillResult.ok({"speed": params.speed, "fingers": fingers})
        except Exception as e:
            return SkillResult.fail(
                f"Failed to set finger positions: {e}",
                {"speed": params.speed, "fingers": fingers},
            )
