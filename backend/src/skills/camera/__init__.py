"""Camera skills for vision and streaming operations."""

from .start_streaming import StartStreamingSkill
from .stop_streaming import StopStreamingSkill
from .get_bounding_box import GetBoundingBoxSkill
from .get_label import GetLabelSkill
from .detect_fast_opencv import DetectFastOpenCVSkill

__all__ = [
    "StartStreamingSkill",
    "StopStreamingSkill",
    "GetBoundingBoxSkill",
    "GetLabelSkill",
    "DetectFastOpenCVSkill",
]
