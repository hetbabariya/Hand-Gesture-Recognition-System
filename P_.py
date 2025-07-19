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

def classify_p_gesture(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1[4]
    index_tip_1 = landmarks1[8]
    thumb_tip_2 = landmarks2[4]
    index_tip_2 = landmarks2[8]
    
    # Calculate the distance threshold
    distance_threshold = 0.1
    
    # Check if the index finger of both hands is touching either hand's thumb
    if (calculate_distance(index_tip_1, thumb_tip_1) < distance_threshold and calculate_distance(index_tip_2, thumb_tip_1) < distance_threshold) or \
       (calculate_distance(index_tip_1, thumb_tip_2) < distance_threshold and calculate_distance(index_tip_2, thumb_tip_2) < distance_threshold):
        return "P"
    else:
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
            
            # Calculate accuracy as the minimum distance between index fingers and thumbs
            index_finger_1_thumb_1_dist = calculate_distance(hand_landmarks1.landmark[4], hand_landmarks1.landmark[8])
            index_finger_2_thumb_2_dist = calculate_distance(hand_landmarks2.landmark[4], hand_landmarks2.landmark[8])
            index_finger_1_thumb_2_dist = calculate_distance(hand_landmarks1.landmark[4], hand_landmarks2.landmark[8])
            index_finger_2_thumb_1_dist = calculate_distance(hand_landmarks2.landmark[4], hand_landmarks1.landmark[8])
            
            min_dist = min(index_finger_1_thumb_1_dist, index_finger_2_thumb_2_dist, index_finger_1_thumb_2_dist, index_finger_2_thumb_1_dist)
            accuracy = 100 - (min_dist * 100)
            
            # Display the accuracy percentage
            cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    # Display the image
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()