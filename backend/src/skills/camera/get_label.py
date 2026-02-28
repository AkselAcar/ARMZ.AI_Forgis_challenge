"""GetLabel skill for GPT-4V OCR / label reading."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class GetLabelParams(BaseModel):
    """Parameters for the get_label skill."""

    prompt: str = Field(
        default="Read any text visible in the image and return it exactly as written.",
        min_length=1,
        description="Instruction prompt for GPT-4V (what to read/extract)",
    )
    use_bbox: bool = Field(
        default=True,
        description="If True, crop to last detected bounding box before sending to GPT-4V",
    )
    crop_margin: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Margin around bbox as fraction of bbox size (only if use_bbox=True)",
    )


@register_skill
class GetLabelSkill(Skill[GetLabelParams]):
    """Read text/labels from image using GPT-4V (OpenAI Vision)."""

    name = "get_label"
    executor_type = "camera"
    description = "Use GPT-4V to read text or labels from the camera image"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return GetLabelParams

    async def validate(self, params: GetLabelParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: GetLabelParams, context: ExecutionContext
    ) -> SkillResult:

        camera_executor = context.get_executor("camera")

        if not camera_executor.is_ready():
            return SkillResult.fail("Camera not connected")

        # Check if we have a bbox to crop to (if requested)
        if params.use_bbox and camera_executor.get_last_bbox() is None:
            return SkillResult.fail(
                "use_bbox=True but no previous detection available. "
                "Run get_bounding_box first or set use_bbox=False."
            )

        result = await camera_executor.read_label(
            prompt=params.prompt,
            use_bbox=params.use_bbox,
            crop_margin=params.crop_margin,
        )

        if result["success"]:
            return SkillResult.ok({
                "label": result["label"],
                "used_bbox": params.use_bbox and camera_executor.get_last_bbox() is not None,
            })
        else:
            return SkillResult.fail(
                result.get("error", "Failed to read label"),
                {"label": ""},
            )
