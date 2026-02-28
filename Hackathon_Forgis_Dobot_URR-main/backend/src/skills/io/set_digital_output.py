"""SetDigitalOutput skill (I/O executor version)."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class IOSetDigitalOutputParams(BaseModel):
    """Parameters for the io_set_digital_output skill."""

    pin: int = Field(
        ...,
        ge=0,
        le=7,
        description="Digital output pin number (0-7)",
    )
    value: bool = Field(
        ...,
        description="Output value (true=HIGH, false=LOW)",
    )


@register_skill
class IOSetDigitalOutputSkill(Skill[IOSetDigitalOutputParams]):
    """Set a digital output pin via I/O executor."""

    name = "io_set_digital_output"
    executor_type = "io_robot"
    description = "Set a digital output pin to HIGH or LOW (via I/O executor)"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return IOSetDigitalOutputParams

    async def validate(
        self, params: IOSetDigitalOutputParams
    ) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: IOSetDigitalOutputParams, context: ExecutionContext
    ) -> SkillResult:
        io_executor = context.get_executor()

        try:
            await io_executor.set_digital_output(params.pin, params.value)
            return SkillResult.ok({"pin": params.pin, "value": params.value})
        except Exception as e:
            return SkillResult.fail(
                f"Failed to set digital output: {e}",
                {"pin": params.pin, "value": params.value},
            )
