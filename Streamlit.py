import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import mediapipe as mp
import numpy as np
import cv2

st.title("Hand Gesture Detection")
st.write("Allow camera access to start gesture detection")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Gesture detection class
class GestureDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

    def process(self, frame):
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame for hand detection
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                
                # Get landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y])
                
                # Detect gestures
                if landmarks[4][1] > landmarks[3][1]:      # Thumb is down
                    cv2.putText(frame, "THUMBS DOWN", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif landmarks[4][1] < landmarks[3][1]:    # Thumb is up
                    cv2.putText(frame, "THUMBS UP", (10, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Check if hand is open or closed
                if (landmarks[8][1] < landmarks[6][1] and    # Index finger up
                    landmarks[12][1] < landmarks[10][1] and  # Middle finger up
                    landmarks[16][1] < landmarks[14][1] and  # Ring finger up
                    landmarks[20][1] < landmarks[18][1]):    # Pinky up
                    cv2.putText(frame, "OPEN HAND", (10, 90), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
        return frame

# Video processor
class VideoProcessor:
    def __init__(self):
        self.detector = GestureDetector()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)  # Mirror image
        
        # Process frame
        img = self.detector.process(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Create webrtc streamer
webrtc_streamer(
    key="gesture_detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("Instructions:")
st.write("1. Allow camera access when prompted")
st.write("2. Show your hand to the camera")
st.write("3. Try these gestures:")
st.write("   - Thumbs up")
st.write("   - Thumbs down")
st.write("   - Open hand (all fingers extended)")