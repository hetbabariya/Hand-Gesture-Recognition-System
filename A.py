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

# Function to check if all fingers except the thumb are closed
def fingers_closed(landmarks, threshold=0.1):
    finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]  # Tips of index, middle, ring, and pinky fingers
    finger_mcp = [landmarks[i] for i in [5, 9, 13, 17]]    # MCP joints of index, middle, ring, and pinky fingers

    return all(calculate_distance(tip, mcp) < threshold for tip, mcp in zip(finger_tips, finger_mcp))

# Function to classify the "A" gesture
def classify_a_gesture(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1[4]
    thumb_tip_2 = landmarks2[4]

    # Calculate distance between the thumb tips of both hands
    distance_threshold = 0.15  # Adjust this threshold as needed
    distance = calculate_distance(thumb_tip_1, thumb_tip_2)

    # Check if the thumb tips of both hands are touching and all other fingers are closed
    if distance < distance_threshold and fingers_closed(landmarks1) and fingers_closed(landmarks2):
        return "A"
    return "Not A"

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
            # Classify the A gesture
            gesture = classify_a_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
