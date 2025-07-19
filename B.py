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

# Function to classify the "B" gesture
def classify_b_gesture(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1[4]
    index_tip_1 = landmarks1[8]
    middle_tip_1 = landmarks1[12]
    ring_tip_1 = landmarks1[16]
    pinki_tip_1 = landmarks1[20]

    thumb_tip_2 = landmarks2[4]
    index_tip_2 = landmarks2[8]
    middle_tip_2 = landmarks2[12]
    ring_tip_2 = landmarks2[16]
    pinki_tip_2 = landmarks2[20]

    # Calculate distances between thumbs and index fingers
    distance_threshold = 0.2 # You can adjust this threshold based on your preference
    distance_1_1 = calculate_distance(thumb_tip_1, index_tip_1)
    distance_1_2 = calculate_distance(thumb_tip_1, middle_tip_1)
    distance_1_3 = calculate_distance(thumb_tip_1, ring_tip_1)
    distance_1_4 = calculate_distance(thumb_tip_1, pinki_tip_1)

    distance_2_1 = calculate_distance(thumb_tip_2, index_tip_2)
    distance_2_2 = calculate_distance(thumb_tip_2, middle_tip_2)
    distance_2_3 = calculate_distance(thumb_tip_2, ring_tip_2)
    distance_2_4 = calculate_distance(thumb_tip_2, pinki_tip_2)
    
    # Check if both thumbs are touching their corresponding index fingers
    if distance_1_1 < distance_threshold and distance_1_2 < distance_threshold and distance_1_3 < distance_threshold and distance_1_4 < distance_threshold and distance_2_1 < distance_threshold and distance_2_2 < distance_threshold and distance_2_3 < distance_threshold and distance_2_4 < distance_threshold:
        distance_between_hands = calculate_distance(thumb_tip_1, thumb_tip_2)
        if distance_between_hands < distance_threshold:
            return "B"
    
    return "Not B"

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
            # Classify the "B" gesture
            gesture = classify_b_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
