"""ROS 2 node that bridges WebSocket camera frames to ROS topics."""

import asyncio
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 1  # Always use freshest frame


class CameraBridgeNode(Node):
    """
    ROS 2 node that receives camera frames from a WebSocket server
    and publishes them as ROS Image messages.

    This enables streaming from a Windows RealSense camera to the
    Docker backend via WebSocket, then republishing to the standard
    ROS camera topic that CameraNode subscribes to.
    """

    def __init__(self):
        super().__init__("camera_bridge_node")

        # Configuration from environment
        self.ws_host = os.environ.get("CAMERA_BRIDGE_HOST", "host.docker.internal")
        self.ws_port = int(os.environ.get("CAMERA_BRIDGE_PORT", "8765"))
        self.ws_uri = f"ws://{self.ws_host}:{self.ws_port}"

        # QoS matching RealSense defaults (BEST_EFFORT)
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Publisher for raw camera images
        self.image_pub = self.create_publisher(
            Image, "/camera/camera/color/image_raw", qos
        )

        # Connection state
        self._connected = False
        self._reconnect_delay = 2.0
        self._running = True

        # Frame queue for dropping old frames
        self._frame_queue: Optional[asyncio.Queue] = None

        # Start WebSocket client in background thread
        self._ws_thread = threading.Thread(target=self._run_ws_client, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(
            f"CameraBridgeNode initialized — connecting to {self.ws_uri}"
        )

    def _run_ws_client(self) -> None:
        """Run the WebSocket client in its own asyncio event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_client_loop())
        except Exception as e:
            self.get_logger().error(f"WebSocket client loop error: {e}")
        finally:
            loop.close()

    async def _ws_client_loop(self) -> None:
        """Main WebSocket client loop with auto-reconnect."""
        try:
            import websockets
        except ImportError:
            self.get_logger().error(
                "websockets package not installed. Run: pip install websockets"
            )
            return

        while self._running:
            try:
                self.get_logger().info(f"Connecting to camera server at {self.ws_uri}")
                async with websockets.connect(self.ws_uri) as websocket:
                    self._connected = True
                    self.get_logger().info("Connected to camera server")

                    # Create bounded queue and start processor
                    self._frame_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
                    processor_task = asyncio.create_task(self._process_loop())

                    try:
                        async for message in websocket:
                            if not self._running:
                                break
                            # Drop oldest frame if queue is full
                            if self._frame_queue.full():
                                try:
                                    self._frame_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    pass
                            try:
                                self._frame_queue.put_nowait(message)
                            except asyncio.QueueFull:
                                pass  # Skip frame
                    finally:
                        processor_task.cancel()
                        try:
                            await processor_task
                        except asyncio.CancelledError:
                            pass

            except websockets.exceptions.ConnectionClosed:
                self.get_logger().warn("Camera server connection closed")
            except ConnectionRefusedError:
                self.get_logger().warn(
                    f"Camera server not available at {self.ws_uri}",
                    throttle_duration_sec=10.0,
                )
            except Exception as e:
                self.get_logger().error(f"WebSocket error: {e}")
            finally:
                self._connected = False
                self._frame_queue = None

            if self._running:
                self.get_logger().info(
                    f"Reconnecting in {self._reconnect_delay} seconds..."
                )
                await asyncio.sleep(self._reconnect_delay)

    async def _process_loop(self) -> None:
        """Process frames from the queue."""
        while self._running and self._frame_queue:
            try:
                jpeg_data = await asyncio.wait_for(
                    self._frame_queue.get(), timeout=0.033
                )
                self._process_frame(jpeg_data)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.get_logger().error(f"Process loop error: {e}")

    def _process_frame(self, jpeg_data: bytes) -> None:
        """Decode JPEG and publish as ROS Image message."""
        try:
            # Decode JPEG to numpy array
            np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                self.get_logger().warn("Failed to decode JPEG frame")
                return

            # Convert BGR to RGB for ROS
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Create ROS Image message
            msg = Image()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_color_optical_frame"
            msg.height, msg.width = frame_rgb.shape[:2]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = msg.width * 3
            msg.data = frame_rgb.tobytes()

            self.image_pub.publish(msg)

            # Log periodically
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count % 100 == 0:
                self.get_logger().info(f"Published {self._frame_count} frames ({msg.width}x{msg.height})")

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def is_connected(self) -> bool:
        """Check if connected to the camera server."""
        return self._connected

    def destroy_node(self) -> None:
        """Clean up resources."""
        self._running = False
        super().destroy_node()
