import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate the distance between two landmarks
def calculate_distance(landmark1, landmark2):
    try:
        return math.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)
    except Exception as e:
        print(f"Error in calculate_distance: {e}")
        return float('inf')

# Function to classify the "B" gesture
def classify_b_gesture(landmarks1, landmarks2):
    try:
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

        # Calculate distances between thumbs and index fingers
        distance_threshold = calculate_distance(landmarks1[1],landmarks1[2])/1.25  # Adjust this threshold as needed
        distance_1_1 = calculate_distance(thumb_tip_1, index_tip_1)
        distance_1_2 = calculate_distance(thumb_tip_1, middle_tip_1)
        distance_1_3 = calculate_distance(thumb_tip_1, ring_tip_1)
        distance_1_4 = calculate_distance(thumb_tip_1, pinky_tip_1)

        distance_2_1 = calculate_distance(thumb_tip_2, index_tip_2)
        distance_2_2 = calculate_distance(thumb_tip_2, middle_tip_2)
        distance_2_3 = calculate_distance(thumb_tip_2, ring_tip_2)
        distance_2_4 = calculate_distance(thumb_tip_2, pinky_tip_2)

        # Initialize a list to track correctness of landmarks
        correct1 = [1] * 21
        correct2 = [1] * 21    

        

        # Check if both thumbs are touching their corresponding index fingers
        if distance_1_1 < distance_threshold and distance_1_2 < distance_threshold and distance_1_3 < distance_threshold and distance_1_4 < distance_threshold and distance_2_1 < distance_threshold and distance_2_2 < distance_threshold and distance_2_3 < distance_threshold and distance_2_4 < distance_threshold:
            distance_between_hands = calculate_distance(thumb_tip_1, thumb_tip_2)
            if distance_between_hands < distance_threshold:
                a = sum(correct1) + sum(correct2)
                accuracy = (a / 42) * 100                
                return "B", correct1, correct2, accuracy

   
        else:
            if not distance_1_1 < distance_threshold:
                for i in range(6, 9):
                    correct1[i] = 0  # Index tip of hand 1
            if not distance_1_2 < distance_threshold:
                for i in range(10, 13):
                    correct1[i] = 0  # Middle tip of hand 1
            if not distance_1_3 < distance_threshold:
                for i in range(14, 17):
                    correct1[i] = 0  # Ring tip of hand 1
            if not distance_1_4 < distance_threshold:
                for i in range(18, 21):
                    correct1[i] = 0  # Pinky tip of hand 1
            if not distance_2_1 < distance_threshold:
                for i in range(6, 9):
                    correct2[i] = 0  # Index tip of hand 2
            if not distance_2_2 < distance_threshold:
                for i in range(10, 13):
                    correct2[i] = 0  # Middle tip of hand 2
            if not distance_2_3 < distance_threshold:
                for i in range(14, 17):
                    correct2[i] = 0  # Ring tip of hand 2
            if not distance_2_4 < distance_threshold:
                for i in range(18, 21):
                    correct2[i] = 0  # Pinky tip of hand 2
            if not calculate_distance(thumb_tip_1, thumb_tip_2) < 0.1:
                for i in range(2, 5):
                    correct1[i] = 0
                    correct2[i] = 0
            a = sum(correct1) + sum(correct2)
            accuracy = (a / 42) * 100 
            return "Not B", correct1, correct2, accuracy
    except Exception as e:
        print(f"Error in classify_b_gesture: {e}")
        return "Error", [0] * 21, [0] * 21, 0

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks1, landmarks2, correct1, correct2):
    try:
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
    except Exception as e:
        print(f"Error in draw_landmarks: {e}")

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)

# Main loop for capturing video and processing hand gestures
while cap.isOpened():
    try:
        # Read a frame from the video capture
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and detect the hands
        results = hands.process(image_rgb)

        # Draw the hand annotations on the image and classify the 'B' gesture
        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                hand_landmarks1 = results.multi_hand_landmarks[0]
                hand_landmarks2 = results.multi_hand_landmarks[1]

                # Classify the B gesture
                gesture, correct1, correct2, accuracy = classify_b_gesture(hand_landmarks1.landmark, hand_landmarks2.landmark)
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
    except Exception as e:
        print(f"Error in main loop: {e}")

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
