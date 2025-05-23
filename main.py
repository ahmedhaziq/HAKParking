# Imports
import cv2
from ultralytics import YOLO

# Load Pretrained Model
cars = YOLO("Models/CarDetectionModel/train2/weights/best.pt")
spots = YOLO("Models/SpotDetectionModel/train6/weights/best.pt")

# Set confidence threshold
CONFIDENCE_THRESHOLD_CARS = 0.98
CONFIDENCE_THRESHOLD_SPOTS = 0.95


# Instantiate video capture
cap = cv2.VideoCapture("Videos/video3.mp4")

# Check if the video is properly opened
if not cap.isOpened():
    print("Error: Cannot open video.")
    exit()

# Process every frame in the video
while True:
    
    # Read the frames and check validity
    ret, frame = cap.read()
    if not ret: break

    # Resize video
    frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)

    # Run object detection
    car = cars(frame)[0]
    spot = spots(frame)[0]
    
    # Loop through results of detection
    for box in car.boxes:
        conf = float(box.conf[0]) # Get confidence values
        
        if conf > CONFIDENCE_THRESHOLD_CARS: # Check for confidence 
            cls = int(box.cls[0]) # Get classification
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Box bounds

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) # Draw rectangle
            cv2.putText(frame, f"{cars.names[cls] } {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Write text
    
    for box in spot.boxes:
        conf = float(box.conf[0])
        
        if conf > CONFIDENCE_THRESHOLD_SPOTS:
            cls = int(box.cls[0]) # Get classification
            
            color = (0,0,0)
            
            if spots.names[cls] == "free_parking_space": color = (0,255,0)
            elif spots.names[cls] == "not_free_parking_space": color = (0,0,255)
            else: color = (0, 165, 255)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # Box bounds

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Draw rectangle
            cv2.putText(frame, f"{spots.names[cls] } {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # Write text

    # Display Frame
    cv2.imshow("Filtered Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break # Exit condition

cap.release()
cv2.destroyAllWindows()
