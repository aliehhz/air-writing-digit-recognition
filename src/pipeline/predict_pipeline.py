import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp

# Load trained model
model = keras.models.load_model('artifacts/model')

def predict_digit(image):
    """ Takes an image (28x28), preprocesses it, and predicts the digit. """
    try:
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize to 28x28
        image = cv2.resize(image, (28, 28))

        # Normalize and reshape
        image = image / 255.0
        image = np.expand_dims(image, axis=(0, -1))  # (1, 28, 28, 1) for CNN

        # Predict
        prediction = model.predict(image)
        digit = np.argmax(prediction)

        return digit
    except Exception as e:
        return f"Error: {str(e)}"
