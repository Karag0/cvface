import cv2
import mediapipe as mp
import math

# Initialize MediaPipe for hand and face tracking
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize OpenCV for webcam capture
cap = cv2.VideoCapture(0)

# Create a window and set it to fullscreen
cv2.namedWindow('Hand and Face Tracking', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand and Face Tracking', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the image horizontally
    frame = cv2.flip(frame, 1)  # 1 = horizontal flip

    # Convert color format for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(frame_rgb)
    results_face = face_mesh.process(frame_rgb)

    # Draw hand tracking
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw face and eye tracking
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Draw facial landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Get eye coordinates (left and right)
            left_eye_indices = [33, 133, 160, 158, 157, 154]  # Left eye indices
            right_eye_indices = [362, 263, 387, 385, 384, 381]  # Right eye indices

            # Convert coordinates to pixels
            h, w, _ = frame.shape
            left_eye_coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in left_eye_indices]
            right_eye_coords = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in right_eye_indices]

            # Draw eyes
            for coord in left_eye_coords:
                cv2.circle(frame, coord, 2, (0, 255, 0), -1)
            for coord in right_eye_coords:
                cv2.circle(frame, coord, 2, (0, 255, 0), -1)

            # Calculate angle between eyes
            if len(left_eye_coords) > 0 and len(right_eye_coords) > 0:
                left_eye_center = left_eye_coords[0]  # First point of left eye
                right_eye_center = right_eye_coords[0]  # First point of right eye
                angle = math.degrees(math.atan2(right_eye_center[1] - left_eye_center[1],
                                                right_eye_center[0] - left_eye_center[0]))

                # Change text color to green and use a cleaner font
                cv2.putText(frame, f'Angle: {angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display result
    cv2.imshow('Hand and Face Tracking', frame)

    # Exit on 'Q' key press
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
