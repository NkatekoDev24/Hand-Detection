import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
import cv2
import mediapipe as mp
import numpy as np

# Page config
st.set_page_config(page_title="Hand Gesture Detection")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Streamlit UI
st.title("Hand Gesture Detection")
st.write("Once you allow camera access, you should see the webcam feed below.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    detection_confidence = st.slider("Detection Confidence", 0.0, 1.0, 0.5)
    tracking_confidence = st.slider("Tracking Confidence", 0.0, 1.0, 0.5)

# Status indicator
status_placeholder = st.empty()

class GestureDetector(VideoProcessorBase):
    def __init__(self) -> None:
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Flip the image for selfie view
        img = cv2.flip(img, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2)
                )
                
                # Get landmark positions
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                
                # Calculate distances for gesture detection
                distance_thumb_index = np.sqrt(
                    (thumb_tip.x - index_tip.x)**2 + 
                    (thumb_tip.y - index_tip.y)**2
                )
                
                # Basic gesture detection
                if thumb_tip.y < index_tip.y:  # Thumb above index
                    if distance_thumb_index > 0.1:
                        gesture = "Thumbs Up"
                    else:
                        gesture = "Closed Hand"
                else:  # Thumb below index
                    if distance_thumb_index > 0.1:
                        gesture = "Thumbs Down"
                    else:
                        gesture = "Pointing"
                
                # Draw gesture text
                cv2.putText(
                    img, 
                    gesture, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create WebRTC streamer
try:
    webrtc_ctx = webrtc_streamer(
        key="gesture-detection",
        mode=webrtc_streamer.WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=GestureDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if webrtc_ctx.state.playing:
        status_placeholder.success("Webcam is running! Make gestures to see them detected.")
    else:
        status_placeholder.info("Click the 'START' button to begin.")
except Exception as e:
    st.error(f"Error accessing webcam: {str(e)}")
    st.info("Please make sure to allow camera access when prompted by your browser.")