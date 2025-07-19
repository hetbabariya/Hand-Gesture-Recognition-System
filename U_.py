import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

def classify_u_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    little_tip = landmarks[20]

    # Check if index finger is open
    index_base = landmarks[5]
    index_pip = landmarks[6]
    index_dip = landmarks[7]
    if (index_pip.y < index_base.y and index_dip.y < index_pip.y):
        index_open = True
    else:
        index_open = False

    # Check if middle, ring, or little finger is above thumb or index finger
    if middle_tip.y < thumb_tip.y or middle_tip.y < index_tip.y or \
       ring_tip.y < thumb_tip.y or ring_tip.y < index_tip.y or \
       little_tip.y < thumb_tip.y or little_tip.y < index_tip.y:
        return "Not U", 0.0

    elif index_open:
        # Calculate the accuracy based on the distance between the thumb and index finger tips
        thumb_index_distance = calculate_distance(thumb_tip, index_tip)
        ideal_distance = 0.2  # Adjust this value based on your preference
        accuracy = max(0.0, 1.0 - abs(thumb_index_distance - ideal_distance) / ideal_distance)
        return "U", accuracy

    else:
        return "Not U", 0.0

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
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Classify the 'u' gesture
        gesture, accuracy = classify_u_gesture(hand_landmarks.landmark)
        cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the accuracy percentage
        cv2.putText(image, f'Accuracy: {accuracy * 100:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()