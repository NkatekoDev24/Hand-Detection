import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# Initialize MediaPipe hands and drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Streamlit app title and description
st.title("Real-Time Hand Gesture Detection")
st.text("The webcam feed will start automatically when you allow camera access.")

# Streamlit sidebar for control
st.sidebar.title("Settings")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
)

# WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def __init__(self):
        self.hands = mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Flip the image horizontally for a mirror view
        img = cv2.flip(img, 1)
        
        # Process the image
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected_image = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if detected_image.multi_hand_landmarks:
            for hand_landmarks in detected_image.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2)
                )

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

        return av.VideoFrame.from_ndarray(image, format="bgr24")

webrtc_streamer(
    key="hand-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)