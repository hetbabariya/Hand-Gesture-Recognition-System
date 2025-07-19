import cv2
import mediapipe as mp
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Define a threshold distance to determine if landmarks are touching
distance_threshold = 0.1

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5

# Function to classify the "Q" gesture
def classify_q_gesture(landmarks1, landmarks2):
    if (calculate_distance(landmarks1[8], landmarks1[4]) < distance_threshold and
        calculate_distance(landmarks1[2], landmarks2[7]) < distance_threshold) or \
       (calculate_distance(landmarks2[8], landmarks2[4]) < distance_threshold and
        calculate_distance(landmarks2[2], landmarks1[7]) < distance_threshold):
        return "Q"
    return "Not Q"

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
        
        # Classify the Q gesture
        gesture = classify_q_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
        cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    else:
        gesture = "Not Q"
        cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
