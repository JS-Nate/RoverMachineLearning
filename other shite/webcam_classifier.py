import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO messages

import cv2
from tensorflow.keras.models import load_model  # Import from TensorFlow
from PIL import Image, ImageOps
import numpy as np


# Step 1: Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
with open("labels.txt", "r") as f:
    class_names = f.readlines()

# Prepare the data shape for the model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Step 2: Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Step 3: Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to a PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Resize the image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert the image to a numpy array and normalize
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the data array
    data[0] = normalized_image_array

    # Step 4: Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Remove any trailing newline
    confidence_score = prediction[0][index]

    # Step 5: Display the results
    cv2.putText(frame, f'Class: {class_name} ({confidence_score:.2f})',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Webcam', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
