"""DetectFastOpenCV skill — high-speed OpenCV box detection."""

from typing import Optional

from pydantic import BaseModel

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class DetectFastOpenCVParams(BaseModel):
    """No parameters needed — ROI is configured inside the executor."""
    pass


@register_skill
class DetectFastOpenCVSkill(Skill[DetectFastOpenCVParams]):
    """Fast OpenCV-based box detection using brightness threshold in a fixed ROI."""

    name = "detect_fast_opencv"
    executor_type = "camera"
    description = "Detect a box on the conveyor using fast OpenCV thresholding"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return DetectFastOpenCVParams

    async def validate(self, params: DetectFastOpenCVParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: DetectFastOpenCVParams, context: ExecutionContext
    ) -> SkillResult:
        camera_executor = context.get_executor("camera")

        if not camera_executor.is_ready():
            return SkillResult.ok({"detected": False})

        detected = await camera_executor.detect_fast_opencv()
        return SkillResult.ok({"detected": detected})
