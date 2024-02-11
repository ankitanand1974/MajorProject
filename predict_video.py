import os
import cv2
from ultralytics import YOLO

# Load YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)

# Set threshold for object detection
threshold = 0.5

# Initialize video capture from default camera
cap = cv2.VideoCapture(0)

# Loop to continuously process frames
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Perform object detection on the frame
    results = model(frame)[0]

    # Process detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        # Check if detection confidence score is above threshold
        if score > threshold:
            # Draw bounding box around the detected object
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Add label of the detected object
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Display processed frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Check for key press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
