import cv2
import mediapipe as mp
import math

# Initialize MediaPipe for hand and face tracking
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize OpenCV for video capture from the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)  # Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440) # Height
# Set FPS (for example, 30 frames per second)
cap.set(cv2.CAP_PROP_FPS, 30)

# Create a window and set it to fullscreen mode
cv2.namedWindow('Hand and Face Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand and Face Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the image
    frame = cv2.flip(frame, 1)  # 1 - mirror horizontally

    # Convert color for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    # Display hand tracking
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display face and eye tracking
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Draw facial landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Get eye coordinates (left and right)
            left_eye_indices = [33, 133, 160, 158, 157, 154]  # Indices for the left eye
            right_eye_indices = [362, 263, 387, 385, 384, 381]  # Indices for the right eye

            # Convert coordinates to pixels
            h, w, _ = frame.shape
            left_eye_coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in
                               left_eye_indices]
            right_eye_coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in
                                right_eye_indices]

            # Draw eyes
            for coord in left_eye_coords:
                cv2.circle(frame, coord, 2, (0, 255, 0), -1)
            for coord in right_eye_coords:
                cv2.circle(frame, coord, 2, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Hand and Face Tracking', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
