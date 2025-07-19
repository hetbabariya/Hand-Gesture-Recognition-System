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

# Function to classify the "D" gesture
def classify_d_gesture(landmarks1, landmarks2):
    # Calculate distances for the conditions
    distance_threshold = 0.1  # Adjust this threshold as needed

    # Check if index tips of both hands are touching and base of index finger of one hand is touching thumb tip of the other hand
    def check_condition(hand1, hand2):
        index_tip_1 = hand1[8]
        index_tip_2 = hand2[8]
        base_index_1 = hand1[5]
        thumb_tip_2 = hand2[4]

        index_tips_touching = calculate_distance(index_tip_1, index_tip_2) < distance_threshold
        thumb_index_touching = calculate_distance(base_index_1, thumb_tip_2) < distance_threshold

        return index_tips_touching and thumb_index_touching

    # Check both possible combinations of hands
    if check_condition(landmarks1, landmarks2) or check_condition(landmarks2, landmarks1):
        return "D"
    return "Not D"

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
            # Classify the D gesture
            gesture = classify_d_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, 'Gesture: Not D', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
