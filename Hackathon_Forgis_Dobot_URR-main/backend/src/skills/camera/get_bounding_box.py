"""GetBoundingBox skill for YOLO object detection."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class GetBoundingBoxParams(BaseModel):
    """Parameters for the get_bounding_box skill."""

    object_class: str = Field(
        ...,
        min_length=1,
        description="Object class to detect (e.g., 'bottle', 'person', 'cup')",
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for detection",
    )


@register_skill
class GetBoundingBoxSkill(Skill[GetBoundingBoxParams]):
    """Detect objects using YOLOv8 and return bounding boxes."""

    name = "get_bounding_box"
    executor_type = "camera"
    description = "Run YOLO object detection and return bounding box for specified class"

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return GetBoundingBoxParams

    async def validate(self, params: GetBoundingBoxParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: GetBoundingBoxParams, context: ExecutionContext
    ) -> SkillResult:
        camera_executor = context.get_executor("camera")

        if not camera_executor.is_ready():
            return SkillResult.fail("Camera not connected")

        detections = await camera_executor.detect_objects(
            class_name=params.object_class,
            confidence_threshold=params.confidence_threshold,
        )

        if not detections:
            return SkillResult.ok({
                "found": False,
                "bbox": None,
                "confidence": 0.0,
                "object_class": params.object_class,
            })

        # Return the first (highest confidence) detection
        best = detections[0]
        return SkillResult.ok({
            "found": True,
            "bbox": {
                "x": best.x,
                "y": best.y,
                "width": best.width,
                "height": best.height,
            },
            "confidence": best.confidence,
            "object_class": best.class_name,
            "total_detections": len(detections),
        })
