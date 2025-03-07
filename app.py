import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs

import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, Response, jsonify
from src.pipeline.predict_pipeline import predict_digit  # Import ML model

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create a blank canvas
canvas = np.zeros((480, 640), dtype=np.uint8)
drawing = False  # Toggle for drawing mode

def gen():
    global drawing, canvas
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Flip camera
        h, w, _ = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get fingertip position
                index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

                if drawing:  # Draw only when mode is enabled
                    cv2.circle(canvas, (x, y), 8, 255, -1)

                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Mark fingertip

        # Merge the camera frame and canvas
        frame_with_canvas = cv2.addWeighted(frame, 0.5, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR), 0.5, 0)

        _, jpeg = cv2.imencode('.jpg', frame_with_canvas)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    cap.release()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/toggle_drawing", methods=["POST"])
def toggle_drawing():
    global drawing
    drawing = not drawing  # Toggle drawing mode
    return "Drawing Mode: " + ("ON" if drawing else "OFF")


@app.route("/clear_canvas", methods=["POST"])
def clear_canvas():
    global canvas
    canvas.fill(0)  # Clear the drawing
    return "Canvas Cleared"



@app.route("/predict", methods=["POST"])
def predict():
    global canvas

    # Pass the raw canvas image to predict_digit (canvas shape is (480, 640))
    digit = int(predict_digit(canvas))
    
    # Optionally, clear the canvas after recognition
    canvas.fill(0)

    return jsonify({"digit": digit})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use PORT from Render, default to 5000 for local runs
    app.run(host="0.0.0.0", port=port, debug=False)
