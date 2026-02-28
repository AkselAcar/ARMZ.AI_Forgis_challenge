"""Camera executor for YOLO detection, OpenAI Vision, and frame streaming."""

import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from openai import AsyncAzureOpenAI

from .base import Executor

if TYPE_CHECKING:
    from api.websocket import WebSocketManager
    from nodes.camera_node import CameraNode

logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Detected object bounding box."""

    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_name: str

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
            "class_name": self.class_name,
        }


class CameraExecutor(Executor):
    """
    Executor for camera operations including YOLO detection and OpenAI Vision.

    Provides streaming, object detection, and OCR capabilities.
    """

    executor_type = "camera"

    def __init__(self, camera_node: "CameraNode", ws_manager: "WebSocketManager"):
        self._camera = camera_node
        self._ws = ws_manager

        # Streaming state
        self._streaming = False
        self._stream_task: Optional[asyncio.Task] = None
        self._send_task: Optional[asyncio.Task] = None
        self._frame_queue: Optional[asyncio.Queue] = None

        # YOLO model (lazy loaded)
        self._yolo_model = None
        self._yolo_model_name = os.environ.get("YOLO_MODEL", "/app/weights/roboflow_logistics.pt")

        # Azure OpenAI client (lazy loaded)
        self._openai_client: Optional[AsyncAzureOpenAI] = None
        self._azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

        # Last detection result for cropping
        self._last_bbox: Optional[BoundingBox] = None

    async def initialize(self) -> None:
        """Wait for camera connection and pre-load YOLO model."""
        logger.info("CameraExecutor initializing...")

        # # Pre-load YOLO model in background to avoid delay on first detection
        # loop = asyncio.get_event_loop()
        # await loop.run_in_executor(None, self._get_yolo_model)

        # Pre-initialize Azure OpenAI client AND warm up the HTTP connection
        # so the first real read_label() call doesn't pay the TLS/auth cold-start
        await self._warmup_openai()

        timeout = 10.0
        elapsed = 0.0
        while not self._camera.has_frame() and elapsed < timeout:
            await asyncio.sleep(0.1)
            elapsed += 0.1

        if not self._camera.has_frame():
            logger.warning("Camera frame not available after timeout - camera may not be connected")
        else:
            dims = self._camera.get_frame_dimensions()
            logger.info(f"CameraExecutor ready - frame size: {dims}")

    async def shutdown(self) -> None:
        """Stop streaming and cleanup."""
        await self.stop_streaming()

    def is_ready(self) -> bool:
        """Check if camera has received frames."""
        return self._camera.has_frame()

    def _get_yolo_model(self):
        """Lazy-load YOLO model on first use."""
        if self._yolo_model is None:
            from ultralytics import YOLO

            logger.info(f"Loading YOLO model: {self._yolo_model_name}")
            self._yolo_model = YOLO(self._yolo_model_name)
            logger.info("YOLO model loaded")
        return self._yolo_model

    def _get_openai_client(self) -> AsyncAzureOpenAI:
        """Get or create Azure OpenAI client."""
        if self._openai_client is None:
            self._openai_client = AsyncAzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        return self._openai_client

    async def _warmup_openai(self) -> None:
        """Send a minimal request to establish the HTTP/TLS connection pool.

        This eliminates the cold-start latency on the first real read_label() call.
        """
        try:
            client = self._get_openai_client()
            await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._azure_deployment,
                    messages=[{"role": "user", "content": "hi"}],
                    max_completion_tokens=1,
                ),
                timeout=15.0,
            )
            logger.info("Azure OpenAI connection warmed up successfully")
        except Exception as e:
            logger.warning(f"Azure OpenAI warmup failed (non-fatal): {e}")

    # --- Streaming ---

    async def start_streaming(self, fps: int = 15, max_queue: int = 1) -> bool:
        """
        Start streaming camera frames over WebSocket.

        Args:
            fps: Target frames per second.
            max_queue: Maximum frames to queue before dropping old ones.

        Returns:
            True if streaming started, False if already streaming.
        """
        if self._streaming:
            logger.warning("Streaming already active")
            return False

        self._streaming = True
        self._frame_queue = asyncio.Queue(maxsize=max_queue)
        self._stream_task = asyncio.create_task(self._capture_loop(fps))
        self._send_task = asyncio.create_task(self._send_loop())
        logger.info(f"Camera streaming started at {fps} FPS (queue size: {max_queue})")
        return True

    async def stop_streaming(self) -> bool:
        """
        Stop streaming camera frames.

        Returns:
            True if streaming stopped, False if not streaming.
        """
        if not self._streaming:
            return False

        self._streaming = False

        for task in [self._stream_task, self._send_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._stream_task = None
        self._send_task = None
        self._frame_queue = None

        logger.info("Camera streaming stopped")
        return True

    def is_streaming(self) -> bool:
        """Check if currently streaming."""
        return self._streaming

    async def _capture_loop(self, fps: int) -> None:
        """Capture frames and put them in the queue, dropping old ones if full."""
        interval = 1.0 / fps

        while self._streaming:
            try:
                frame_jpeg = self._camera.get_frame_jpeg(quality=70)
                if frame_jpeg and self._frame_queue:
                    dims = self._camera.get_frame_dimensions()
                    width, height = dims if dims else (640, 480)

                    frame_data = {
                        "frame": base64.b64encode(frame_jpeg).decode(),
                        "width": width,
                        "height": height,
                    }

                    # Drop oldest frame if queue is full
                    if self._frame_queue.full():
                        try:
                            self._frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass

                    try:
                        self._frame_queue.put_nowait(frame_data)
                    except asyncio.QueueFull:
                        pass  # Skip this frame

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Capture error: {e}")
                await asyncio.sleep(interval)

    async def _send_loop(self) -> None:
        """Send frames from the queue to WebSocket clients."""
        while self._streaming:
            try:
                if self._frame_queue:
                    frame_data = await asyncio.wait_for(
                        self._frame_queue.get(), timeout=0.033
                    )
                    await self._ws.broadcast("camera_frame", frame_data)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Send error: {e}")

    # --- Object Detection ---

    async def detect_objects(
        self,
        class_name: Optional[str] = None,
        confidence_threshold: float = 0.5,
    ) -> list[BoundingBox]:
        """
        Run YOLO object detection on current frame.

        Args:
            class_name: Filter by class name (e.g., "bottle", "person").
            confidence_threshold: Minimum confidence threshold.

        Returns:
            List of detected bounding boxes.
        """
        frame = self._camera.get_latest_frame()
        if frame is None:
            logger.warning("No frame available for detection")
            return []

        logger.info(f"detect_objects: frame shape={frame.shape}, looking for class='{class_name}' conf>={confidence_threshold}")

        # Run detection on full frame
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self._get_yolo_model()(frame, verbose=False),
        )

        detections: list[BoundingBox] = []
        model = self._get_yolo_model()

        for result in results:
            logger.info(f"detect_objects: YOLO returned {len(result.boxes)} raw boxes")
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                logger.info(f"detect_objects: raw box class='{cls_name}' conf={conf:.3f} xyxy=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

                # Filter by confidence
                if conf < confidence_threshold:
                    logger.info(f"detect_objects: SKIPPED (conf {conf:.3f} < {confidence_threshold})")
                    continue

                # Filter by class name if specified
                if class_name and cls_name.lower() != class_name.lower():
                    logger.info(f"detect_objects: SKIPPED (class '{cls_name}' != '{class_name}')")
                    continue

                bbox = BoundingBox(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=conf,
                    class_name=cls_name,
                )
                logger.info(f"detect_objects: ACCEPTED bbox x={bbox.x:.1f} y={bbox.y:.1f} w={bbox.width:.1f} h={bbox.height:.1f}")
                detections.append(bbox)

        # Store last detection for potential cropping
        if detections:
            self._last_bbox = detections[0]
            logger.info(f"detect_objects: broadcasting bbox to frontend: {detections[0].to_dict()}")
            try:
                await self._broadcast_bbox(detections[0])
                logger.info("detect_objects: bbox broadcast sent successfully")
            except Exception as e:
                logger.error(f"detect_objects: bbox broadcast FAILED: {e}")
        else:
            logger.warning("detect_objects: no detections passed filters, no bbox broadcast")

        logger.info(f"detect_objects: returning {len(detections)} detections")
        return detections

    async def _broadcast_bbox(self, bbox: BoundingBox) -> None:
        """Broadcast bounding box to frontend for overlay display."""
        dims = self._camera.get_frame_dimensions()
        width, height = dims if dims else (640, 480)
        payload = {
            "bbox": bbox.to_dict(),
            "frame_width": width,
            "frame_height": height,
            "display_duration_ms": 5000,
        }
        logger.info(f"_broadcast_bbox: sending payload frame={width}x{height} bbox={bbox.to_dict()}")
        await self._ws.broadcast("bounding_box", payload)

    def get_last_bbox(self) -> Optional[BoundingBox]:
        """Get the last detected bounding box."""
        return self._last_bbox

    # --- OpenAI Vision / OCR ---

    async def read_label(
        self,
        prompt: str,
        use_bbox: bool = True,
        crop_margin: float = 0.1,
    ) -> dict:
        """
        Use GPT-4V to read text/labels from the image.

        Args:
            prompt: Instruction for what to read (e.g., "Read the product label").
            use_bbox: If True and last detection exists, crop to that region.
            crop_margin: Margin around bbox as fraction of bbox size.

        Returns:
            Dict with 'label' (extracted text) and 'success' status.
        """
        # Brief delay to let the box settle under the camera
        await asyncio.sleep(1.5)

        frame = self._camera.get_latest_frame()
        if frame is None:
            return {"success": False, "label": "", "error": "No frame available"}

        # Crop to fixed pick zone (125x125 centered at frame_center + 50px right, + 75px down)
        frame, _, _ = self._crop_to_pick_zone(frame)

        # Optionally crop further to last detected bbox
        # if use_bbox and self._last_bbox:
            # frame = self._crop_to_bbox(frame, self._last_bbox, margin=crop_margin)

        # Save debug image to see what is sent to the API
        cv2.imwrite("/app/debug_label_crop.jpg", frame)
        logger.info(f"read_label: saved debug crop to /app/debug_label_crop.jpg (shape={frame.shape})")

        # Encode frame to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return {"success": False, "label": "", "error": "Failed to encode image"}

        image_bytes = encoded.tobytes()
        b64_image = base64.b64encode(image_bytes).decode()

        try:
            client = self._get_openai_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._azure_deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=300,
                ),
                timeout=50.0,  # generous timeout to handle cold-start connection
            )

            label = response.choices[0].message.content or ""
            logger.info(f"OpenAI Vision response: {label[:100]}...")
            return {"success": True, "label": label.strip()}

        except asyncio.TimeoutError:
            logger.error("OpenAI Vision timeout after 25 seconds")
            return {"success": False, "label": "", "error": "OpenAI Vision timeout"}
        except Exception as e:
            logger.error(f"OpenAI Vision error: {e}")
            return {"success": False, "label": "", "error": str(e)}

    async def check_quality(
        self,
        prompt: str,
        use_bbox: bool = False,
        crop_margin: float = 0.1,
    ) -> dict:
        """
        Use GPT-4V to check if a label is readable.

        Args:
            prompt: Instruction for quality check (should ask for READABLE/NOT_READABLE).
            use_bbox: If True and last detection exists, crop to that region.
            crop_margin: Margin around bbox as fraction of bbox size.

        Returns:
            Dict with 'readable' (bool) and 'success' status.
        """
        # Brief delay to let the box settle under the camera
        await asyncio.sleep(1.5)

        frame = self._camera.get_latest_frame()
        if frame is None:
            return {"success": False, "readable": False, "error": "No frame available"}

        # Crop to fixed pick zone
        frame, _, _ = self._crop_to_pick_zone(frame)

        # Save debug image
        cv2.imwrite("/app/debug_qc_crop.jpg", frame)
        logger.info(f"check_quality: saved debug crop to /app/debug_qc_crop.jpg (shape={frame.shape})")

        # Encode frame to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
        success, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not success:
            return {"success": False, "readable": False, "error": "Failed to encode image"}

        image_bytes = encoded.tobytes()
        b64_image = base64.b64encode(image_bytes).decode()

        try:
            client = self._get_openai_client()
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=self._azure_deployment,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=50,
                ),
                timeout=50.0,
            )

            raw_response = response.choices[0].message.content or ""
            raw_response = raw_response.strip().upper()
            logger.info(f"check_quality OpenAI response: {raw_response}")

            # Parse response - look for READABLE or NOT_READABLE
            readable = "READABLE" in raw_response and "NOT_READABLE" not in raw_response

            return {
                "success": True,
                "readable": readable,
                "raw_response": raw_response,
            }

        except asyncio.TimeoutError:
            logger.error("check_quality: OpenAI Vision timeout")
            return {"success": False, "readable": False, "error": "OpenAI Vision timeout"}
        except Exception as e:
            logger.error(f"check_quality: OpenAI Vision error: {e}")
            return {"success": False, "readable": False, "error": str(e)}

    def _crop_to_pick_zone(self, frame: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Crop frame to the fixed pick zone (300x350 at center + 70px right, + 15px down).

        Returns:
            Tuple of (cropped_frame, offset_x, offset_y) where offsets map
            cropped coordinates back to the original frame.
        """
        h, w = frame.shape[:2]
        cx = w // 2 + 70
        cy = h // 2 + 15
        crop_w = 300
        crop_h = 350

        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)

        return frame[y1:y2, x1:x2], x1, y1

    def _crop_to_bbox(
        self,
        frame: np.ndarray,
        bbox: BoundingBox,
        margin: float = 0.1,
    ) -> np.ndarray:
        """Crop frame to bounding box with margin."""
        h, w = frame.shape[:2]

        # Add margin
        margin_x = bbox.width * margin
        margin_y = bbox.height * margin

        x1 = max(0, int(bbox.x - margin_x))
        y1 = max(0, int(bbox.y - margin_y))
        x2 = min(w, int(bbox.x + bbox.width + margin_x))
        y2 = min(h, int(bbox.y + bbox.height + margin_y))

        return frame[y1:y2, x1:x2]

    # --- Snapshot ---

    def get_snapshot_jpeg(self, quality: int = 90) -> Optional[bytes]:
        """Get current frame as JPEG for REST endpoint."""
        return self._camera.get_frame_jpeg(quality=quality)

    def get_state_summary(self) -> dict:
        """Get camera state summary."""
        dims = self._camera.get_frame_dimensions()
        return {
            "connected": self._camera.has_frame(),
            "streaming": self._streaming,
            "frame_size": {"width": dims[0], "height": dims[1]} if dims else None,
            "last_detection": self._last_bbox.to_dict() if self._last_bbox else None,
        }
