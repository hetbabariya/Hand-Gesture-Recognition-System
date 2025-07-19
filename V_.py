import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

def calculate_angle(landmark1, landmark2, landmark3):
    """
    Calculate the angle between three landmarks based on the dot product
    """
    vec1 = [landmark1.x - landmark2.x, landmark1.y - landmark2.y]
    vec2 = [landmark3.x - landmark2.x, landmark3.y - landmark2.y]
    
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    if magnitude1 * magnitude2 == 0:
        return 0
    
    angle = math.acos(dot_product / (magnitude1 * magnitude2))
    return angle * 180 / math.pi  # Convert to degrees

def classify_v_gesture(landmarks):
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    
    # Calculate the angle between index and middle fingers
    angle = calculate_angle(index_tip, middle_tip, landmarks[6])  # Use the wrist landmark as the pivot
    
    # Calculate accuracy based on the angle
    target_angle = 30  # Adjust this value based on your requirements
    accuracy = 100 - abs(angle - target_angle) / target_angle * 100
    accuracy = max(0, min(100, accuracy))  # Clamp accuracy between 0 and 100
    
    if accuracy >= 50:
        return "V", accuracy
    else:
        return "Not V", accuracy

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

            # Classify the 'V' gesture and get the accuracy
            gesture, accuracy = classify_v_gesture(hand_landmarks.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()