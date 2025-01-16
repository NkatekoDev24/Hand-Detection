import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

capture = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    while capture.isOpened():
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected_image = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if detected_image.multi_hand_landmarks:
            for hand_landmarks in detected_image.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                                          mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))

                # Extracting specific landmarks for hand gesture recognition
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Calculate distance between thumb and index finger tip
                distance_thumb_index = ((thumb_tip.x - index_finger_tip.x) ** 2 +
                                        (thumb_tip.y - index_finger_tip.y) ** 2) ** 0.5

                # Check if thumb tip is above index finger tip
                thumb_above_index = thumb_tip.y < index_finger_tip.y

                # Check if thumb tip is below index finger tip
                thumb_below_index = thumb_tip.y > index_finger_tip.y

                # Check if all fingers are extended
                all_fingers_extended = (distance_thumb_index > 0.1 and
                                        distance_thumb_index > 0.1 and
                                        distance_thumb_index > 0.1 and
                                        distance_thumb_index > 0.1)

                # Recognize a thumbs-up gesture if the hand is a fist with an open thumb facing up
                if all_fingers_extended and thumb_above_index:
                    cv2.putText(image, 'Thumbs Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Recognize a thumbs-down gesture if the hand is a fist with an open thumb facing down
                elif all_fingers_extended and thumb_below_index:
                    cv2.putText(image, 'Thumbs Down', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Recognize a fist gesture if the distance between thumb and index finger is below a threshold
                elif distance_thumb_index < 0.05:  # You can adjust this threshold as needed
                    cv2.putText(image, 'Fist', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # Recognize an open hand if all fingers are extended
                elif all_fingers_extended:
                    cv2.putText(image, 'Open Hand', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

capture.release()
cv2.destroyAllWindows()
