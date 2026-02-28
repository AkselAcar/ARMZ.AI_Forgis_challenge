"""REST endpoints for camera control."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/camera", tags=["camera"])

# CameraExecutor will be injected via app state
_camera_executor = None


def set_camera_executor(executor) -> None:
    """Set the camera executor instance (called during app initialization)."""
    global _camera_executor
    _camera_executor = executor


def get_executor():
    """Get the camera executor, raising if not initialized."""
    if _camera_executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera executor not initialized",
        )
    return _camera_executor


# --- Request/Response Models ---


class StreamStartRequest(BaseModel):
    """Request to start streaming."""

    fps: int = Field(default=15, ge=1, le=30, description="Target FPS")


class StreamResponse(BaseModel):
    """Response for stream control operations."""

    success: bool
    streaming: bool
    message: Optional[str] = None


class CameraStateResponse(BaseModel):
    """Response for camera state."""

    connected: bool
    streaming: bool
    frame_size: Optional[dict] = None
    last_detection: Optional[dict] = None


# --- Endpoints ---


@router.post("/stream/start", response_model=StreamResponse)
async def start_stream(request: StreamStartRequest = StreamStartRequest()):
    """Start streaming camera frames over WebSocket."""
    executor = get_executor()

    if not executor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera not connected",
        )

    started = await executor.start_streaming(fps=request.fps)

    return StreamResponse(
        success=True,
        streaming=True,
        message=f"Streaming started at {request.fps} FPS" if started else "Streaming already active",
    )


@router.post("/stream/stop", response_model=StreamResponse)
async def stop_stream():
    """Stop streaming camera frames."""
    executor = get_executor()

    stopped = await executor.stop_streaming()

    return StreamResponse(
        success=True,
        streaming=False,
        message="Streaming stopped" if stopped else "Streaming was not active",
    )


@router.get("/snapshot")
async def get_snapshot(quality: int = 90):
    """
    Get a single JPEG snapshot from the camera.

    Args:
        quality: JPEG quality (1-100).

    Returns:
        JPEG image data.
    """
    executor = get_executor()

    if not executor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera not connected",
        )

    jpeg_data = executor.get_snapshot_jpeg(quality=min(100, max(1, quality)))

    if jpeg_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No frame available",
        )

    return Response(
        content=jpeg_data,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=snapshot.jpg"},
    )


@router.get("/state", response_model=CameraStateResponse)
async def get_camera_state():
    """Get current camera state."""
    executor = get_executor()
    return executor.get_state_summary()
