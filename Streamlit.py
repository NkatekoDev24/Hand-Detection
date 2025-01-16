# file_path: app.py

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time

# Initialize MediaPipe hands and drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Streamlit app title and description
st.title("Real-Time Hand Gesture Detection")
st.text("Press 'Start' to activate the webcam and detect hand gestures.")

# Streamlit sidebar for control
st.sidebar.title("Settings")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

# Initialize session state for webcam control
if "run" not in st.session_state:
    st.session_state.run = False

# Streamlit buttons for webcam control
start_webcam = st.button("Start Webcam")
stop_webcam = st.button("Stop Webcam")

if start_webcam:
    st.session_state.run = True

if stop_webcam:
    st.session_state.run = False

# Placeholder for video output
frame_window = st.empty()

# Webcam input processing
if st.session_state.run:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        st.error("Webcam not accessible!")
        st.session_state.run = False

    with mp_hands.Hands(min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence) as hands:
        while st.session_state.run:
            ret, frame = capture.read()
            if not ret:
                st.error("Failed to capture webcam frame.")
                break

            # Flip the image horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_image = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if detected_image.multi_hand_landmarks:
                for hand_landmarks in detected_image.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))

                    # Extract landmarks
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    distance_thumb_index = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                                            (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
                    thumb_above_index = thumb_tip.y < index_finger_tip.y
                    thumb_below_index = thumb_tip.y > index_finger_tip.y

                    # Recognize gestures
                    if distance_thumb_index > 0.1 and thumb_above_index:
                        cv2.putText(image, 'Thumbs Up', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    elif distance_thumb_index > 0.1 and thumb_below_index:
                        cv2.putText(image, 'Thumbs Down', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    elif distance_thumb_index < 0.05:
                        cv2.putText(image, 'Fist', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(image, 'Open Hand', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Convert BGR image to RGB for Streamlit
            frame_window.image(image, channels="BGR")
            time.sleep(0.03)  # Add a small delay to reduce CPU usage

    capture.release()
    cv2.destroyAllWindows()
