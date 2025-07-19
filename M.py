import cv2
import mediapipe as mp

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Function to check if a point is above the line formed by two other points
def is_point_above_line(point, line_start, line_end):
    # Calculate the direction vectors
    line_vector = (line_end.x - line_start.x, line_end.y - line_start.y)
    point_vector = (point.x - line_start.x, point.y - line_start.y)
    
    # Cross product to determine the relative position
    return (point_vector[0] * line_vector[1] - point_vector[1] * line_vector[0]) > 0

# Function to classify the "M" gesture
def classify_m_gesture(landmarks1, landmarks2):
    wrist = landmarks2[17]
    palm_base = landmarks2[0]

    # Points of index, middle, and ring fingers
    index_finger = landmarks1[8]
    middle_finger = landmarks1[12]
    ring_finger = landmarks1[16]
    
    # Check if the fingers are above the line between wrist and palm base
    if is_point_above_line(index_finger, wrist, palm_base) and \
       is_point_above_line(middle_finger, wrist, palm_base) and \
       is_point_above_line(ring_finger, wrist, palm_base):
        return "Not M"
    return "M"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the hands
    results = hands.process(image_rgb)
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand_landmarks1 = results.multi_hand_landmarks[0]
        hand_landmarks2 = results.multi_hand_landmarks[1]
        
        mp_drawing.draw_landmarks(image, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, hand_landmarks2, mp_hands.HAND_CONNECTIONS)
        
        # Classify the M gesture
        gesture = classify_m_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
        cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        gesture = "Not M"
        cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
