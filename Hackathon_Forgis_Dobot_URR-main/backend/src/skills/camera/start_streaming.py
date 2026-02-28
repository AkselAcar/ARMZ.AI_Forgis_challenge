"""StartStreaming skill for starting WebSocket camera frame broadcast."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class StartStreamingParams(BaseModel):
    """Parameters for the start_streaming skill."""

    fps: int = Field(
        default=15,
        ge=1,
        le=30,
        description="Target frames per second for streaming",
    )


@register_skill
class StartStreamingSkill(Skill[StartStreamingParams]):
    """Start streaming camera frames over WebSocket."""

    name = "start_streaming"
    executor_type = "camera"
    description = "Start streaming camera frames over WebSocket at specified FPS"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return StartStreamingParams

    async def validate(self, params: StartStreamingParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: StartStreamingParams, context: ExecutionContext
    ) -> SkillResult:
        camera_executor = context.get_executor("camera")

        if not camera_executor.is_ready():
            return SkillResult.fail("Camera not connected")

        started = await camera_executor.start_streaming(fps=params.fps)

        if started:
            return SkillResult.ok({"streaming": True, "fps": params.fps})
        else:
            return SkillResult.fail(
                "Streaming already active",
                {"streaming": True, "fps": params.fps},
            )
