# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ultralytics",
#     "opencv-python",
#     "pyrealsense2",
# ]
# ///
"""Test YOLO model detection on RealSense camera or image."""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Test YOLO model detection")
    parser.add_argument(
        "--model",
        default="backend/weights/roboflow_logistics.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--image",
        help="Path to test image (if not provided, uses RealSense camera)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)

    # Print available classes
    print(f"\nAvailable classes ({len(model.names)}):")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    if args.image:
        # Single image mode
        print(f"\nRunning detection on: {args.image}")
        results = model(args.image, conf=args.confidence)

        for result in results:
            print(f"\nDetections:")
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"  - {cls_name}: {conf:.2%} at [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")

        # Show annotated image
        annotated = results[0].plot()
        cv2.imshow("Detection Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # RealSense camera mode
        import pyrealsense2 as rs

        print(f"\nStarting RealSense camera ({args.width}x{args.height})...")
        print("Press 'q' to quit, 's' to save screenshot")

        # Configure RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.rgb8, 30)

        try:
            pipeline.start(config)
            print("RealSense camera started")
        except Exception as e:
            print(f"Error: Could not start RealSense camera: {e}")
            return

        try:
            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert to numpy array (RGB)
                frame_rgb = np.asanyarray(color_frame.get_data())
                # Convert to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # Run detection
                results = model(frame_bgr, conf=args.confidence, verbose=False)

                # Draw results
                annotated = results[0].plot()

                # Show detection count
                detections = []
                for box in results[0].boxes:
                    cls_id = int(box.cls)
                    detections.append(model.names[cls_id])

                info_text = f"Detections: {len(detections)}"
                if detections:
                    info_text += f" | {', '.join(detections)}"
                cv2.putText(annotated, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("YOLO Detection (press 'q' to quit)", annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite("detection_screenshot.jpg", annotated)
                    print("Screenshot saved: detection_screenshot.jpg")
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
