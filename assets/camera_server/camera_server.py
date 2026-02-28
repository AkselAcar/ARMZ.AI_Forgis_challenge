"""
Windows RealSense Camera WebSocket Server.

Captures frames from a RealSense camera and streams them as JPEG over WebSocket.
Run this script on Windows with the RealSense camera connected.

Usage:
    python camera_server.py --port 8765 --fps 30 --width 640 --height 480
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RealSenseCapture:
    """Captures frames from a RealSense camera."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline: Optional[rs.pipeline] = None
        self.config: Optional[rs.config] = None

    def start(self) -> bool:
        """Initialize and start the RealSense pipeline."""
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.config.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
            )
            self.pipeline.start(self.config)
            logger.info(
                f"RealSense started: {self.width}x{self.height} @ {self.fps} FPS"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to start RealSense: {e}")
            return False

    def stop(self) -> None:
        """Stop the RealSense pipeline."""
        if self.pipeline:
            try:
                self.pipeline.stop()
                logger.info("RealSense stopped")
            except Exception as e:
                logger.error(f"Error stopping RealSense: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera."""
        if not self.pipeline:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None


class CameraServer:
    """WebSocket server that streams camera frames to connected clients."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 80,
    ):
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.camera = RealSenseCapture(width=width, height=height, fps=fps)
        self.clients: set = set()
        self.running = False
        self._frame_interval = 1.0 / fps

    async def register_client(self, websocket) -> None:
        """Register a new client and keep connection alive."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        self.clients.add(websocket)
        try:
            # Keep connection open, wait for close
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {client_addr}")

    async def broadcast_frames(self) -> None:
        """Capture and broadcast frames to all connected clients."""
        while self.running:
            if not self.clients:
                await asyncio.sleep(0.1)
                continue

            frame = self.camera.get_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Convert RGB to BGR for OpenCV encoding
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Encode as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, jpeg_data = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not success:
                continue

            jpeg_bytes = jpeg_data.tobytes()

            # Use websockets.broadcast for thread-safe sending
            websockets.broadcast(self.clients, jpeg_bytes)

            await asyncio.sleep(self._frame_interval)

    async def start(self) -> None:
        """Start the camera and WebSocket server."""
        if not self.camera.start():
            logger.error("Failed to start camera, exiting")
            return

        self.running = True

        # Start frame broadcast task
        broadcast_task = asyncio.create_task(self.broadcast_frames())

        # Start WebSocket server with ping disabled to avoid race conditions
        logger.info(f"WebSocket server starting on ws://{self.host}:{self.port}")
        try:
            async with websockets.serve(
                self.register_client,
                self.host,
                self.port,
                ping_interval=None,  # Disable ping to avoid race with broadcast
            ):
                await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            self.camera.stop()

    def stop(self) -> None:
        """Signal the server to stop."""
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="RealSense Camera WebSocket Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument(
        "--quality", type=int, default=80, help="JPEG quality (0-100)"
    )
    args = parser.parse_args()

    server = CameraServer(
        host=args.host,
        port=args.port,
        fps=args.fps,
        width=args.width,
        height=args.height,
        jpeg_quality=args.quality,
    )

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()
