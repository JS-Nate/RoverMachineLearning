import cv2
import numpy as np
import math


# Function to calculate the angle between two vectors in degrees
def calculate_angle(v1, v2):
    # Compute the dot product of the vectors
    dot_product = np.dot(v1, v2)
    # Compute the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    # Calculate the cosine of the angle using the dot product formula
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    # Ensure the value is within the valid range [-1, 1] to avoid numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    # Return the angle in degrees
    return np.degrees(np.arccos(cos_theta))


# Define the target colors (RGB) and their names
colors = {
    'Earth': (119, 122, 129),  # RGB for Earth
    'Sun': (175, 95, 114),  # RGB for Sun
    'Venus': (174, 126, 120),  # RGB for Venus
    'Jupiter': (129, 233, 121)  # RGB for Jupiter
}

# Define the tolerance (range) for each color (in RGB space)
color_thresholds = 40  # Allows a range of 40 units in each RGB channel

# Initialize the camera
cap = cv2.VideoCapture(1)  # 0 is the default camera

# Angle threshold for deciding if the car needs to turn (2 degrees)
angle_threshold = 2

# Variable to store the name of the currently tracked object
currently_tracking = None

# List to store the objects that have been collected
collected_objects = []

while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the dimensions of the frame (height, width)
    height, width, _ = frame.shape

    # The bottom middle of the camera screen
    bottom_center = (width // 2, height)

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Set lower and upper bounds for brightness (Value channel in HSV)
    lower_brightness = np.array([0, 0, 50])  # Low brightness threshold
    upper_brightness = np.array([179, 255, 255])  # Max brightness (full color range)

    # Create a mask for areas with non-black colors (low Value areas are excluded)
    mask_non_black = cv2.inRange(hsv, lower_brightness, upper_brightness)

    # Find contours of the non-black regions
    contours, _ = cv2.findContours(mask_non_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []  # List to store the objects' bounding box, centroid, and name

    # Process each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            # Extract the region of interest (ROI)
            roi = frame[y:y + h, x:x + w]

            # Calculate the average color of the object in RGB
            avg_color_bgr = np.mean(roi, axis=(0, 1))  # BGR format
            avg_color_rgb = (int(avg_color_bgr[2]), int(avg_color_bgr[1]), int(avg_color_bgr[0]))  # Convert to RGB

            # Find the closest match for the color
            detected_name = None
            min_distance = float('inf')
            for name, target_color in colors.items():
                # Compute the Euclidean distance between the target color and the average color
                distance = np.linalg.norm(np.array(target_color) - np.array(avg_color_rgb))

                # If the distance is within the threshold, label the object with the name
                if distance < min_distance and distance < color_thresholds:
                    min_distance = distance
                    detected_name = name

            # Only add the object if it's detected and labeled
            if detected_name:
                # Calculate the centroid of the bounding box (center of the object)
                centroid = (x + w // 2, y + h // 2)
                objects.append((centroid, detected_name, avg_color_rgb))

                # Draw the bounding box around the detected object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green bounding box

                # Put the color label above the bounding box
                label = f"{detected_name}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, avg_color_rgb, 2, cv2.LINE_AA)

    # If we have detected objects, create a path between them
    if len(objects) > 0:
        # Start from the bottom center
        remaining_objects = objects.copy()
        current_point = bottom_center
        first_turn_displayed = False  # Flag to ensure we only label the first line

        # Find the closest object to the bottom center
        closest_obj = min(remaining_objects, key=lambda obj: np.linalg.norm(np.array(current_point) - np.array(obj[0])))
        remaining_objects.remove(closest_obj)

        # Draw the line from the current point to the closest object
        cv2.line(frame, current_point, closest_obj[0], (0, 255, 255), 2)  # Yellow line to the object

        # Update the tracking object
        if currently_tracking is None:
            currently_tracking = closest_obj[1]

        # Display the current tracking object
        cv2.putText(frame, f"Currently Tracking: {currently_tracking}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 255), 2, cv2.LINE_AA)

        # Calculate the vector representing the current line (from bottom_center to the closest object)
        current_vector = np.array(closest_obj[0]) - np.array(current_point)
        # Vertical vector is straight up from the bottom center (0, -1)
        vertical_vector = np.array([0, -1])

        # Calculate the angle between the current vector and the vertical vector
        angle = calculate_angle(current_vector, vertical_vector)

        # If the angle exceeds the threshold, display which way to turn
        if angle > angle_threshold and not first_turn_displayed:
            # Reverse the logic: if the x-component is positive, turn right, otherwise turn left
            direction = "Turn Right" if current_vector[0] > 0 else "Turn Left"
            cv2.putText(frame, direction, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            first_turn_displayed = True  # Set flag to ensure labeling happens only once
        # If the angle is small (within the threshold), display "Go Forward"
        elif angle <= angle_threshold and not first_turn_displayed:
            cv2.putText(frame, "Go Forward", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            first_turn_displayed = True  # Set flag to ensure labeling happens only once

        # Update the current point to the closest object's centroid
        current_point = closest_obj[0]

    # Handle key press events for "x"
    key = cv2.waitKey(1) & 0xFF
    if key == ord('x') and currently_tracking is not None:
        collected_objects.append(currently_tracking)
        cv2.putText(frame, f"Collected: {currently_tracking}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        # Mark the object as no longer being tracked
        currently_tracking = None  # Stop tracking the current object

    # Display the original frame with bounding boxes, labels, and path
    cv2.imshow("Non-Black Detection", frame)

    # Press 'q' to quit the application
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
