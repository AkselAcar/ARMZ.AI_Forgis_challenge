import cv2
import numpy as np
import requests
import time

# --- Configuration ---
CAMERA_URL = "http://localhost:8000/api/camera/snapshot"
FLOW_START_URL = "http://localhost:8000/api/flows/step6_fast_conveyor_pick/start"

# Tuned for cardboard brown 
LOWER_BROWN = np.array([10, 50, 50])
UPPER_BROWN = np.array([30, 255, 200])

MIN_BOX_AREA = 4000  # Minimum size to ignore small specks of dust/noise

def get_snapshot():
    try:
        response = requests.get(CAMERA_URL, stream=True, timeout=2)
        if response.status_code == 200:
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            return cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Camera error: {e}")
    return None

def trigger_robot():
    print("🚀 BOX DETECTED ON BELT! Triggering robot...")
    try:
        requests.post(FLOW_START_URL)
        print("Waiting for robot to finish cycle (10 seconds)...")
        time.sleep(10) # Pause the vision loop while the robot moves
    except Exception as e:
        print(f"Failed to start flow: {e}")

if __name__ == "__main__":
    print("👁️ Starting OpenCV Vision Controller with ROI...")
    
    while True:
        frame = get_snapshot()
        
        if frame is None:
            time.sleep(0.5)
            continue
            
        # --- THE FIX: CROP THE IMAGE (REGION OF INTEREST) ---
        # The camera is 640x480. We slice the array to ignore the top 150 pixels.
        # This completely removes the background cardboard and cables from the AI's view.
        roi = frame[150:480, 0:640] 
        
        # 1. Convert ONLY the cropped area to HSV
        hsv_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 2. Create the color mask
        mask = cv2.inRange(hsv_frame, LOWER_BROWN, UPPER_BROWN)
        
        # 3. Find the blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        box_found = False
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > MIN_BOX_AREA:
                # Get coordinates based on the cropped ROI
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw on the original frame (we add 150 to the Y axis to match the original image height)
                cv2.rectangle(frame, (x, y + 150), (x + w, y + 150 + h), (0, 255, 0), 3)
                cv2.putText(frame, "BOX DETECTED", (x, y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                box_found = True
                break

        # Show the live feed (you will see the green box snap onto the cardboard)
        cv2.imshow("Conveyor Belt Vision", frame)
        
        # Show the black and white mask just to help you debug what the AI is seeing!
        cv2.imshow("What the AI Sees (White = Brown)", mask)
        
        if box_found:
            trigger_robot()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()