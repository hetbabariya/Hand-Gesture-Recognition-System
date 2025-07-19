import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Define a threshold distance to determine if landmarks are touching
distance_threshold = 0.15

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2 + (landmark1.z - landmark2.z) ** 2)

# Function to classify the "S" gesture
def classify_s_gesture(landmarks1, landmarks2):
    # List of the landmark indices to be checked
    indices = [18, 19, 20]
    # Check the distances for each corresponding landmark
    for index in indices:
        if calculate_distance(landmarks1[index], landmarks2[index]) >= distance_threshold:
            return "Not S"
    return "S"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the hands
    results = hands.process(image_rgb)
    
    # Initialize the gesture as 'Not S'
    gesture = "Not S"

    # Draw the hand annotations on the image
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand_landmarks1 = results.multi_hand_landmarks[0]
        hand_landmarks2 = results.multi_hand_landmarks[1]
        
        mp_drawing.draw_landmarks(image, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, hand_landmarks2, mp_hands.HAND_CONNECTIONS)
        
        # Classify the S gesture
        gesture = classify_s_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
    
    # Display the gesture result on the frame
    cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
