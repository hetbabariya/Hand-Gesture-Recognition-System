import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

# Function to classify the "F" gesture
def classify_f_gesture(landmarks1, landmarks2):
    distance_threshold = 0.125  # Adjust this threshold as needed

    # Calculate distances for the required landmark pairs
    distances = [
        calculate_distance(landmarks1[11], landmarks2[11]),
        calculate_distance(landmarks1[7], landmarks2[10]),
        calculate_distance(landmarks1[6], landmarks2[6]),
        calculate_distance(landmarks1[10], landmarks2[7])
    ]

    # Check if all required landmark pairs are within the threshold distance
    if all(distance < distance_threshold for distance in distances):
        return "F"
    return "Not F"

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and detect the hands
    results = hands.process(image_rgb)
    
    # Draw the hand annotations on the image
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            hand_landmarks1 = results.multi_hand_landmarks[0]
            hand_landmarks2 = results.multi_hand_landmarks[1]
            mp_drawing.draw_landmarks(image, hand_landmarks1, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, hand_landmarks2, mp_hands.HAND_CONNECTIONS)
            # Classify the F gesture
            gesture = classify_f_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Gesture: Not F', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
