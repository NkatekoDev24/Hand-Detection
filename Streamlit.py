import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av

# Initialize MediaPipe hands and drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Streamlit app title and description
st.title("Real-Time Hand Gesture Detection")
st.text("Allow camera access to start hand gesture detection.")

# Streamlit sidebar for control
st.sidebar.title("Settings")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

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
        
        # Convert to RGB
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detected_image = self.hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if detected_image.multi_hand_landmarks:
            for hand_landmarks in detected_image.multi_hand_landmarks:
                # Draw landmarks
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
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Calculate distances and positions
                distance_thumb_index = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                                      (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5
                
                thumb_above_index = thumb_tip.y < index_finger_tip.y
                thumb_below_index = thumb_tip.y > index_finger_tip.y

                # Check if all fingers are extended
                all_fingers_extended = (
                    distance_thumb_index > 0.1 and
                    middle_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
                    ring_finger_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
                    pinky_tip.y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
                )

                # Gesture recognition with your original logic
                if all_fingers_extended and thumb_above_index:
                    cv2.putText(image, 'Thumbs Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif all_fingers_extended and thumb_below_index:
                    cv2.putText(image, 'Thumbs Down', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif distance_thumb_index < 0.05:
                    cv2.putText(image, 'Fist', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                elif all_fingers_extended:
                    cv2.putText(image, 'Open Hand', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create WebRTC streamer
webrtc_streamer(
    key="hands",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)