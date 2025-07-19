import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Define reference landmarks for Savasana (Corpse Pose)
reference_pose = {
    "left_ankle": np.array([0.1, 0.9]),  # Left ankle position
    "right_ankle": np.array([0.9, 0.9]),  # Right ankle position
    "left_shoulder": np.array([0.3, 0.4]),  # Left shoulder position
    "right_shoulder": np.array([0.7, 0.4]),  # Right shoulder position
    "head": np.array([0.5, 0.2])  # Head position
}

# Function to calculate the distance between two points
def calculate_distance(p1, p2):
    return np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

# Function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks, correct):
    h, w, _ = image.shape
    for idx, landmark in enumerate(landmarks.landmark):
        color = (0, 255, 0) if correct[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 1, color, -3)
    
    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmarks.landmark[start_idx]
        end_point = landmarks.landmark[end_idx]
        color = (0, 255, 0) if correct[start_idx] == 1 and correct[end_idx] == 1 else (0, 0, 255)
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        cv2.line(image, start, end, color, 1)

def calculate_angle(point1, point2, point3):
    # Calculate vectors
    vec1 = [point1[0] - point2[0], point1[1] - point2[1]]
    vec2 = [point3[0] - point2[0], point3[1] - point2[1]]
    
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

def detect_pose(pose_landmarks):
    if pose_landmarks:
        # Define the detected landmarks for Savasana (Corpse Pose)
        detected_pose = {
            "left_ankle": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]),
            "right_ankle": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]),
            "left_shoulder": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                    pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]),
            "right_shoulder": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]),
            "head": np.array([pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].x,
                            pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y])
        }

        # Set all landmarks as correct initially
        correct = [1] * len(pose_landmarks.landmark)
        feedback = []

        # Check if the hands are touching the ankles
        left_leg_hand_touch = calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST],
                                                 pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]) < calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) * 4
        right_leg_hand_touch = calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST],
                                                  pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]) < calculate_distance(pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP], pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]) * 4

        # Calculate similarity score and mark incorrect landmarks
        total_distance = 0.0
        for key, idx in zip(reference_pose.keys(), range(len(pose_landmarks.landmark))):
            detected_point = detected_pose[key]
            ref_point = reference_pose[key]
            distance = np.linalg.norm(detected_point - ref_point)
            total_distance += distance
            if distance > 0.2:  # Mark as incorrect if distance is greater than 0.2
                correct[idx] = 0
                feedback.append(f"{key.replace('_', ' ').title()} position is incorrect\n")

        # If hands and legs are touching, mark as incorrect
        if left_leg_hand_touch:
            correct[mp_pose.PoseLandmark.LEFT_WRIST.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_ANKLE.value] = 0
            feedback.append("Left hand should not touch left ankle\n")

        if right_leg_hand_touch:
            correct[mp_pose.PoseLandmark.RIGHT_WRIST.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_ANKLE.value] = 0
            feedback.append("Right hand should not touch right ankle\n")

        if pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT.value].y > pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value].y or pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT.value].y > pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value].y:
            correct[mp_pose.PoseLandmark.NOSE.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EYE_INNER.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE_OUTER.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EYE_INNER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_EAR.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_EAR.value] = 0
            correct[mp_pose.PoseLandmark.MOUTH_RIGHT.value] = 0
            correct[mp_pose.PoseLandmark.MOUTH_LEFT.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_HIP.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_HIP.value] = 0
            correct[mp_pose.PoseLandmark.LEFT_SHOULDER.value] = 0
            correct[mp_pose.PoseLandmark.RIGHT_SHOULDER.value] = 0
            feedback.append("Body alignment is incorrect\n")

        # Calculate accuracy based on correct landmarks
        accuracy = sum(correct) / len(correct) * 100
        pose_name = "Savasana (Corpse Pose)" if accuracy == 100 else "None"
        
        feedback_str = ", ".join(feedback) if feedback else "Pose is correct"

        return accuracy, pose_name, correct, feedback_str

    return 0.0, "None", [0] * len(pose_landmarks.landmark), "No pose detected"

# Main function to capture video and detect pose
def main():
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB and process it with MediaPipe Pose
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                accuracy, pose_name, correct, feedback_str = detect_pose(results.pose_landmarks)
                draw_landmarks(frame, results.pose_landmarks, correct)
                cv2.putText(frame, f'Pose: {pose_name}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f'Feedback: {feedback_str}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Pose Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
