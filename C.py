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

# Function to classify the "C" gesture
def classify_c_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    thumb_mcp = landmarks[1]
    index_mcp = landmarks[5]
    middle_mcp = landmarks[9]
    ring_mcp = landmarks[13]
    pinky_mcp = landmarks[17]

    # Calculate distances from the tips to the MCP joints
    distance_threshold = 0.05  # Adjust this threshold as needed
    thumb_open = calculate_distance(thumb_tip, thumb_mcp) > distance_threshold
    index_open = calculate_distance(index_tip, index_mcp) > distance_threshold
    middle_open = calculate_distance(middle_tip, middle_mcp) > distance_threshold
    ring_open = calculate_distance(ring_tip, ring_mcp) > distance_threshold
    pinky_open = calculate_distance(pinky_tip, pinky_mcp) > distance_threshold

    # Check if all fingers and thumb are open
    if thumb_open and index_open and middle_open and ring_open and pinky_open:
        return "C"
    return "Not C"

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Classify the C gesture
            gesture = classify_c_gesture(hand_landmarks.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
