import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Function to calculate the angle between three points
def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vec1 = [point1.x - point2.x, point1.y - point2.y]
    vec2 = [point3.x - point2.x, point3.y - point2.y]
    
    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    # Calculate magnitudes
    mag1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    mag2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Calculate angle in radians
    radians = math.acos(dot_product / (mag1 * mag2))
    
    # Convert radians to degrees
    angle = math.degrees(radians)
    
    return angle

# Function to classify the "L" gesture
def classify_l_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    wrist = landmarks[0]
    
    # Calculate the angle between thumb tip, index tip, and wrist
    angle_threshold = 33  # Adjust this threshold as needed
    angle = calculate_angle(thumb_tip, index_tip, wrist)
    
    # Check if the angle is greater than 90 degrees
    if angle > angle_threshold:
        return "L", angle
    return "Not L", angle

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
            # Classify the L gesture
            gesture, angle = classify_l_gesture(hand_landmarks.landmark)
            cv2.putText(image, f'Gesture: {gesture}, Angle: {angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
