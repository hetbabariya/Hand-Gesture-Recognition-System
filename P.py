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

# Function to classify the "P" gesture
def classify_p_gesture(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1[4]
    index_tip_1 = landmarks1[8]
    middle_tip_1 = landmarks1[12]
    ring_tip_1 = landmarks1[16]
    pinky_tip_1 = landmarks1[20]

    thumb_tip_2 = landmarks2[4]
    index_tip_2 = landmarks2[8]
    middle_tip_2 = landmarks2[12]
    ring_tip_2 = landmarks2[16]
    pinky_tip_2 = landmarks2[20]

    # Calculate distances for the first hand (thumb touching all other fingers)
    distance_threshold = 0.1  # Adjust this threshold as needed
    distances_hand1 = [
        calculate_distance(thumb_tip_1, index_tip_1),
        calculate_distance(thumb_tip_1, middle_tip_1),
        calculate_distance(thumb_tip_1, ring_tip_1),
        calculate_distance(thumb_tip_1, pinky_tip_1)
    ]

    # Calculate distance for the second hand (index finger touching the thumb of the first hand)
    distance_hand2 = calculate_distance(thumb_tip_1, index_tip_2)

    # Calculate distances for the second hand (thumb touching all other fingers)
    distances_hand2 = [
        calculate_distance(thumb_tip_2, index_tip_2),
        calculate_distance(thumb_tip_2, middle_tip_2),
        calculate_distance(thumb_tip_2, ring_tip_2),
        calculate_distance(thumb_tip_2, pinky_tip_2)
    ]

    # Calculate distance for the first hand (index finger touching the thumb of the second hand)
    distance_hand1 = calculate_distance(thumb_tip_2, index_tip_1)

    # Check if all fingers of hand1 are touching its thumb and hand2's index finger is touching hand1's thumb
    if all(distance < distance_threshold for distance in distances_hand1) and distance_hand2 < distance_threshold:
        return "P"
    # Check if all fingers of hand2 are touching its thumb and hand1's index finger is touching hand2's thumb
    if all(distance < distance_threshold for distance in distances_hand2) and distance_hand1 < distance_threshold:
        return "P"

    return "Not P"

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
            # Classify the P gesture
            gesture = classify_p_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
