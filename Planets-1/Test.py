import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best.pt")

# Load video file
cap = cv2.VideoCapture(0)

# Define codec and output video writer for saving detections
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_with_detections.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS),
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection on the frame
    results = model.predict(frame)

    # Annotate frame with detection results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

# Release resources
cap.release()
out.release()

print("Processing complete. Annotated video saved as 'output_with_detections.mp4'.")
