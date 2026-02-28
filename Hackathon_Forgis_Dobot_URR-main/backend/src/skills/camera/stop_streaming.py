"""StopStreaming skill for stopping WebSocket camera frame broadcast."""

from typing import Optional

from pydantic import BaseModel

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class StopStreamingParams(BaseModel):
    """Parameters for the stop_streaming skill (none required)."""

    pass


@register_skill
class StopStreamingSkill(Skill[StopStreamingParams]):
    """Stop streaming camera frames over WebSocket."""

    name = "stop_streaming"
    executor_type = "camera"
    description = "Stop streaming camera frames over WebSocket"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return StopStreamingParams

    async def validate(self, params: StopStreamingParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: StopStreamingParams, context: ExecutionContext
    ) -> SkillResult:
        camera_executor = context.get_executor("camera")

        stopped = await camera_executor.stop_streaming()

        if stopped:
            return SkillResult.ok({"streaming": False})
        else:
            return SkillResult.ok(
                {"streaming": False, "message": "Streaming was not active"}
            )
