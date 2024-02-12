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

# Variables for vehicle counting
vehicle_count = 0
prev_vehicle_count = 0

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
            # Check if the detected object is a vehicle (you might need to adjust the class ID)
            if results.names[int(class_id)] == 'car' or results.names[int(class_id)] == 'truck':
                # Draw bounding box around the detected vehicle
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                # Add label of the detected vehicle
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
                # Check if the vehicle has crossed a threshold line (for simplicity, assume a horizontal line)
                if y1 > 400:  # Adjust this value according to the position of the threshold line
                    vehicle_count += 1

    # Draw the threshold line
    cv2.line(frame, (0, 300), (frame.shape[1], 300), (0, 0, 255), 2)

    # Display vehicle count
    cv2.putText(frame, f'Count: {vehicle_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display processed frame
    cv2.imshow('Real-Time Object Detection', frame)

    # Check for key press 'q' to exit loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
