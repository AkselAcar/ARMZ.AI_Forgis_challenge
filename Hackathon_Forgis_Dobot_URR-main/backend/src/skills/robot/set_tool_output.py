"""SetToolOutput skill — controls a tool digital output (DOBOT ToolDO)."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class SetToolOutputParams(BaseModel):
    """Parameters for the set_tool_output skill."""

    index: int = Field(
        ...,
        ge=1,
        description="ToolDO index (1=close gripper, 2=open gripper on dual-solenoid)",
    )
    status: int = Field(
        default=1,
        ge=0,
        le=1,
        description="Output status (1=activate, 0=deactivate)",
    )


@register_skill
class SetToolOutputSkill(Skill[SetToolOutputParams]):
    """Activate or deactivate a tool digital output via the robot executor."""

    name = "set_tool_output"
    executor_type = "robot"
    description = "Control a DOBOT tool digital output (e.g. pneumatic gripper via ToolDO)"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return SetToolOutputParams

    async def validate(
        self, params: SetToolOutputParams
    ) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: SetToolOutputParams, context: ExecutionContext
    ) -> SkillResult:
        robot_executor = context.get_executor("robot")

        try:
            success = await robot_executor.set_tool_output(params.index, params.status)
            if success:
                return SkillResult.ok({"index": params.index, "status": params.status})
            return SkillResult.fail(
                f"ToolDO(index={params.index}, status={params.status}) returned error",
                {"index": params.index, "status": params.status},
            )
        except Exception as e:
            return SkillResult.fail(
                f"set_tool_output failed: {e}",
                {"index": params.index, "status": params.status},
            )
