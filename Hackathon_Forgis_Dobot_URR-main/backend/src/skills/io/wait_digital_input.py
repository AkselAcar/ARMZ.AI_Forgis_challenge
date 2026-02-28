"""WaitDigitalInput skill for waiting on digital input state."""

import asyncio
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class WaitDigitalInputParams(BaseModel):
    """Parameters for the wait_digital_input skill."""

    pin: int = Field(
        ...,
        ge=0,
        le=7,
        description="Digital input pin number (0-7)",
    )
    expected_value: bool = Field(
        ...,
        description="Expected value to wait for (true=HIGH, false=LOW)",
    )
    poll_interval_ms: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Polling interval in milliseconds",
    )


@register_skill
class WaitDigitalInputSkill(Skill[WaitDigitalInputParams]):
    """Wait for a digital input pin to reach a specific state."""

    name = "wait_digital_input"
    executor_type = "io_robot"
    description = "Wait for a digital input to become HIGH or LOW"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return WaitDigitalInputParams

    async def validate(
        self, params: WaitDigitalInputParams
    ) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: WaitDigitalInputParams, context: ExecutionContext
    ) -> SkillResult:
        io_executor = context.get_executor()
        poll_interval_s = params.poll_interval_ms / 1000.0

        # Note: Timeout is handled by the FlowExecutor at the step level
        while True:
            current_value = await io_executor.get_digital_input(params.pin)

            if current_value is None:
                return SkillResult.fail(
                    f"Digital input pin {params.pin} not available",
                    {"pin": params.pin},
                )

            if current_value == params.expected_value:
                return SkillResult.ok(
                    {
                        "pin": params.pin,
                        "expected_value": params.expected_value,
                        "matched": True,
                    }
                )

            await asyncio.sleep(poll_interval_s)
