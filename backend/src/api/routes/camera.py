"""REST endpoints for camera control."""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])

# CameraExecutor + FlowManager will be injected via app state
_camera_executor = None
_flow_manager = None


def set_camera_executor(executor) -> None:
    """Set the camera executor instance (called during app initialization)."""
    global _camera_executor
    _camera_executor = executor


def set_camera_flow_manager(manager) -> None:
    """Set the flow manager so auto-detect can trigger flows."""
    global _flow_manager
    _flow_manager = manager


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

# ── Auto-detect background loop ─────────────────────────────────────────────

_auto_detect_task: Optional[asyncio.Task] = None
_auto_detect_running = False
_target_flow_id: Optional[str] = None
_POLL_INTERVAL = 0.01          # 100 Hz detection
_COOLDOWN_AFTER_FLOW = 1.0     # seconds to wait after flow completes


async def _continuous_detection_loop():
    """
    Background loop that:
      1. Polls detect_fast_opencv() at ~100 Hz
      2. When box_detected → triggers the target flow via FlowManager
      3. Waits for the flow to finish, then resumes monitoring
    """
    global _auto_detect_running
    executor = get_executor()

    logger.info(
        "Auto-detect loop STARTED  (flow=%s, poll=%.0f Hz)",
        _target_flow_id, 1 / _POLL_INTERVAL,
    )

    while _auto_detect_running:
        if not executor.is_ready():
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        try:
            detected = await executor.detect_fast_opencv()
        except Exception:
            await asyncio.sleep(_POLL_INTERVAL)
            continue

        if detected and _target_flow_id and _flow_manager is not None:
            logger.info(
                "Box detected! Triggering flow '%s' via /box_detection",
                _target_flow_id,
            )
            started, msg = await _flow_manager.start_flow(_target_flow_id)
            if started:
                logger.info("Flow started: %s", msg)
                # Wait for flow to finish before resuming detection
                while _auto_detect_running:
                    st = _flow_manager.get_status()
                    if st.status not in ("running", "paused"):
                        break
                    await asyncio.sleep(0.5)
                logger.info("Flow finished. Cooling down %.1fs…", _COOLDOWN_AFTER_FLOW)
                await asyncio.sleep(_COOLDOWN_AFTER_FLOW)
            else:
                logger.warning("Could not start flow: %s", msg)
                await asyncio.sleep(1.0)   # back-off
        else:
            await asyncio.sleep(_POLL_INTERVAL)

    logger.info("Auto-detect loop STOPPED")


class AutoDetectStartRequest(BaseModel):
    """Request body for starting auto-detect with an optional flow_id."""
    flow_id: Optional[str] = Field(
        default=None,
        description="Flow to trigger when a box is detected. "
                    "If omitted, detection publishes to /box_detection only.",
    )


@router.post("/auto-detect/start")
async def start_auto_detect(body: AutoDetectStartRequest = AutoDetectStartRequest()):
    """
    Start the background camera monitoring loop.

    When a box enters the ROI the result is published to the /box_detection
    ROS topic.  If *flow_id* is provided the corresponding flow is launched
    automatically on each detection.
    """
    global _auto_detect_task, _auto_detect_running, _target_flow_id

    if _auto_detect_running:
        return {"message": "Auto-detection is already running.", "flow_id": _target_flow_id}

    _target_flow_id = body.flow_id
    _auto_detect_running = True
    _auto_detect_task = asyncio.create_task(_continuous_detection_loop())
    return {
        "message": "Auto-detection loop STARTED. Monitoring camera…",
        "flow_id": _target_flow_id,
    }


@router.post("/auto-detect/stop")
async def stop_auto_detect():
    """Stop the background monitoring loop."""
    global _auto_detect_running, _auto_detect_task
    _auto_detect_running = False
    if _auto_detect_task and not _auto_detect_task.done():
        _auto_detect_task.cancel()
        _auto_detect_task = None
    return {"message": "Auto-detection loop STOPPED."}


@router.get("/auto-detect/status")
async def auto_detect_status():
    """Get current auto-detect state."""
    return {
        "running": _auto_detect_running,
        "flow_id": _target_flow_id,
    }