import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

# Function to check if all fingers except the thumb are closed
def fingers_closed(landmarks, threshold=0.1):
    try:
        finger_tips = [landmarks[i] for i in [8, 12, 16, 20]]  # Tips of index, middle, ring, and pinky fingers
        finger_mcp = [landmarks[i] for i in [5, 9, 13, 17]]    # MCP joints of index, middle, ring, and pinky fingers

        return all(calculate_distance(tip, mcp) < threshold for tip, mcp in zip(finger_tips, finger_mcp))
    except IndexError as e:
        print(f"IndexError in fingers_closed: {e}")
        return False

def classify_p_gesture(landmarks1, landmarks2):
    thumb_tip_1 = landmarks1[4]
    index_tip_1 = landmarks1[8]
    thumb_tip_2 = landmarks2[4]
    index_tip_2 = landmarks2[8]

    # Calculate the distance threshold
    distance_threshold = calculate_distance(landmarks1[1],landmarks1[2])/1.25 

    # Initialize a list to track correctness of landmarks
    correct1 = [1] * 21
    correct2 = [1] * 21

    # Calculate accuracy as the minimum distance between index fingers and thumbs
    index_finger_1_thumb_1_dist = calculate_distance(hand_landmarks1.landmark[4], hand_landmarks1.landmark[8])
    index_finger_2_thumb_1_dist = calculate_distance(hand_landmarks2.landmark[4], hand_landmarks1.landmark[8])

    min_dist = min(index_finger_1_thumb_1_dist, index_finger_2_thumb_1_dist)
    accuracy = 100 - (min_dist * 100)


    # Check if the index finger of both hands is touching either hand's thumb
    if (calculate_distance(index_tip_1, thumb_tip_1) < distance_threshold and calculate_distance(index_tip_2, thumb_tip_1) < distance_threshold) :
        a = sum(correct1) + sum(correct2)
        accuracy = (a / 42) * 100 
        if landmarks2[12].y < landmarks2[9].y or landmarks2[16].y < landmarks2[13].y or landmarks2[20].y < landmarks2[17].y or landmarks1[12].y < landmarks1[9].y or landmarks1[16].y < landmarks1[13].y or landmarks1[20].y < landmarks1[17].y:
            if landmarks2[12].y < landmarks2[9].y:
                for i in range(10, 13):
                    correct2[i] = 0  # middle tip of hand 2
            if landmarks2[16].y < landmarks2[13].y:
                for i in range(14, 17):
                    correct2[i] = 0  # ring tip of hand 2
            if landmarks2[20].y < landmarks2[17].y:
                for i in range(18, 21):
                    correct2[i] = 0  # pinky tip of hand 2
            if landmarks1[12].y < landmarks1[9].y:
                for i in range(10, 13):
                    correct1[i] = 0  # middle tip of hand 1
            if landmarks1[16].y < landmarks1[13].y:
                for i in range(14, 17):
                    correct1[i] = 0  # ring tip of hand 1
            if landmarks1[20].y < landmarks1[17].y:
                for i in range(18, 21):
                    correct1[i] = 0  # pinky tip of hand 1
            a = sum(correct1) + sum(correct2)
            accuracy = (a / 42) * 100 
            return "Not P", correct1, correct2 ,accuracy

        return "P", correct1, correct2 ,accuracy
    else:

        if landmarks2[12].y < landmarks2[9].y:
            for i in range(10, 13):
                correct2[i] = 0  # middle tip of hand 2
        if landmarks2[16].y < landmarks2[13].y:
            for i in range(14, 17):
                correct2[i] = 0  # ring tip of hand 2
        if landmarks2[20].y < landmarks2[17].y:
            for i in range(18, 21):
                correct2[i] = 0  # pinky tip of hand 2
        if landmarks1[12].y < landmarks1[9].y:
            for i in range(10, 13):
                correct1[i] = 0  # middle tip of hand 1
        if landmarks1[16].y < landmarks1[13].y:
            for i in range(14, 17):
                correct1[i] = 0  # ring tip of hand 1
        if landmarks1[20].y < landmarks1[17].y:
            for i in range(18, 21):
                correct1[i] = 0  # pinky tip of hand 1

        # Mark relevant landmarks as incorrect
        for i in range(6, 9):
            correct2[i] = 0  # Index tip of hand 2

        for i in range(6, 9):
            correct1[i] = 0  # Index tip of hand 1
    
        for i in range(2, 5):
            correct1[i] = 0  # Thumb tip of hand 1

        a = sum(correct1) + sum(correct2)
        accuracy = (a / 42) * 100 

        return "Not P", correct1, correct2 ,accuracy

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks1, landmarks2, correct1, correct2):
    h, w, _ = image.shape

    # Draw landmarks and connections for hand 1
    connections1 = mp_hands.HAND_CONNECTIONS
    for idx, landmark in enumerate(landmarks1):
        color = (0, 255, 0) if correct1[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)

    for connection in connections1:
        start_idx, end_idx = connection
        start_point = landmarks1[start_idx]
        end_point = landmarks1[end_idx]
        color = (0, 255, 0) if correct1[start_idx] == 1 and correct1[end_idx] == 1 else (0, 0, 255)
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        cv2.line(image, start, end, color, 2)

    # Draw landmarks and connections for hand 2
    connections2 = mp_hands.HAND_CONNECTIONS
    for idx, landmark in enumerate(landmarks2):
        color = (0, 255, 0) if correct2[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)

    for connection in connections2:
        start_idx, end_idx = connection
        start_point = landmarks2[start_idx]
        end_point = landmarks2[end_idx]
        color = (0, 255, 0) if correct2[start_idx] == 1 and correct2[end_idx] == 1 else (0, 0, 255)
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        cv2.line(image, start, end, color, 2)

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Main loop for capturing video and processing hand gestures
while cap.isOpened():
    # Read a frame from the video capture
    success, image = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect the hands
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image and classify the 'P' gesture
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks) == 2:
            hand_landmarks1 = results.multi_hand_landmarks[0]
            hand_landmarks2 = results.multi_hand_landmarks[1]

            # Classify the P gesture
            gesture, correct1, correct2 ,accuracy = classify_p_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw landmarks and connections with colors based on correctness
            draw_landmarks(image, hand_landmarks1.landmark, hand_landmarks2.landmark, correct1, correct2)


            # Display the accuracy percentage
            cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
