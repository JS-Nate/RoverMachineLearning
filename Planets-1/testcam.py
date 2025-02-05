import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/Users/natt4/OneDrive/Documents/GitHub/RoverMachineLearning/Planets-1/best.pt")

# Read the image file
image = cv2.imread("C:/Users/natt4/OneDrive/Documents/GitHub/RoverMachineLearning/Planets-1/IMG_1190.jpg")

# Check if the image file is read correctly
if image is None:
    print("Error: Could not read image file.")
    exit()  # Exit the script

# Perform detection on the image
results = model.predict(image)

# Annotate image with detection results
annotated_image = results[0].plot()

# Resize the window
cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("YOLO Detection", 1280, 720)  # Set the desired window size

# Display the image with detections
cv2.imshow("YOLO Detection", annotated_image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
