import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/Users/natt4/PycharmProjects/RoverMachineLearning/runs/detect/train3/weights/best.pt")

# Open webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model.predict(frame)

    # Annotate frame with detection results
    annotated_frame = results[0].plot()

    # Display the frame with detections
    cv2.imshow("YOLO Detection", annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
