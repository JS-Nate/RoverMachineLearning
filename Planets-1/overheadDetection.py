import torch
import cv2

# Load the YOLOv5 model with custom weights
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Perform object detection
    results = model(frame)
    
    # Render the results on the frame
    frame = results.render()[0]
    
    # Display the resulting frame
    cv2.imshow('Object Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()