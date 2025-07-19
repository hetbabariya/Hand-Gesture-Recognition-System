import cv2  # Import the OpenCV library
import mediapipe as mp  # Import the Mediapipe library
import numpy as np  # Import the NumPy library
import math

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands  # Access the Hands module from Mediapipe
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Initialize the Hands model with parameters
mp_drawing = mp.solutions.drawing_utils  # Access the drawing utilities from Mediapipe

# Initialize OpenCV Video Capture
cap = cv2.VideoCapture(0)  # Initialize video capture from the default camera (index 0)

# Define a function to calculate the 2D angle between four points
def calculate_2d_angle(point1, point2, point3, point4, h, w):
    # Convert Mediapipe landmarks to numpy arrays (2D)
    p1 = np.array([point1.x * w, point1.y * h])
    p2 = np.array([(point2.x * w + point4.x * w) / 2, (point2.y * h + point4.y * h) / 2])
    p3 = np.array([point3.x * w, point3.y * h])
    
    # Calculate vectors
    vec1 = p1 - p2
    vec2 = p3 - p2
    
    # Calculate the angle between vectors
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.arccos(cosine_angle)
    
    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle)
    return angle_degrees

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

# Define a function to classify the 'U' gesture
def classify_v_gesture(landmarks, a1, a2, a3, a4):
    # Extract landmarks for fingers, thumb, and wrist
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    index_dip = landmarks[7]
    index_mcp = landmarks[5]

    middle_tip = landmarks[12]
    middle_dip = landmarks[11]
    middle_pip = landmarks[10]
    middle_mcp = landmarks[9]

    ring_tip = landmarks[16]
    ring_dip = landmarks[15]
    ring_pip = landmarks[14]
    ring_mcp = landmarks[13]
    
    pinky_tip = landmarks[20]
    pinky_dip = landmarks[19]
    pinky_mcp = landmarks[17]
    wrist = landmarks[0]

    # Calculate distances from wrist to finger MCP joints
    w_middle_mcp = wrist.y - middle_mcp.y
    if(w_middle_mcp < 0):
        w_middle_mcp = -w_middle_mcp
    
    w_middle_tip = wrist.y - middle_tip.y
    if(w_middle_tip < 0):
        w_middle_tip = -w_middle_tip

    w_middle_dip = wrist.y - middle_dip.y
    if(w_middle_dip < 0):
        w_middle_dip = -w_middle_dip
    
    w_ring_mcp = wrist.y - ring_mcp.y
    if(w_ring_mcp < 0):
        w_ring_mcp = -w_ring_mcp
    
    w_ring_tip = wrist.y - ring_tip.y
    if(w_ring_tip < 0):
        w_ring_tip = -w_ring_tip

    w_ring_dip = wrist.y - ring_dip.y
    if(w_ring_dip < 0):
        w_ring_dip = -w_ring_dip

    w_pinky_mcp = wrist.y - pinky_mcp.y
    if(w_pinky_mcp < 0):
        w_pinky_mcp = -w_pinky_mcp
    
    w_pinky_tip = wrist.y - pinky_tip.y
    if(w_pinky_tip < 0):
        w_pinky_tip = -w_pinky_tip

    w_pinky_dip = wrist.y - pinky_dip.y
    if(w_pinky_dip < 0):
        w_pinky_dip = -w_pinky_dip

    w_index_tip = wrist.y - index_tip.y
    if(w_index_tip < 0):
        w_index_tip = -w_index_tip

    w_index_dip = wrist.y - index_dip.y
    if(w_index_dip < 0):
        w_index_dip = -w_index_dip

    # Initialize a list to track correctness of landmarks
    correct = [0]*21
    correct[0] = 1
    correct[1] = 1
    correct[5] = 1
    correct[9] = 1
    correct[13] = 1
    correct[17] = 1
    
    # Classify based on angles and distances
    for i in range(6,9):
        correct[i] = 1
        
    if a2 < 160:
        for i in range(2, 5):
            correct[i] = 1
    
    if (w_middle_mcp > w_ring_tip):
        for i in range(14, 17):
            correct[i] = 1
    
    if (w_middle_mcp > w_pinky_tip):
        for i in range(18, 21):
            correct[i] = 1

    if not (landmarks[10].y > w_middle_tip):
            for i in range(10, 13):
                correct[i] = 1

    if (w_index_dip > w_index_tip):
        for i in range(6, 9):
            correct[i] = 0
    
    '''angle_index_finger=calculate_angle(landmarks[7],landmarks[6],landmarks[5])
    if not 175 <= angle_index_finger <= 185 :
        for i in range(6, 9):
            correct[i] = 0
    '''   
    angle_middle_finger=calculate_angle(landmarks[11],landmarks[10],landmarks[9])
    if  175 <= angle_middle_finger <= 185 :
        for i in range(10, 13):
            correct[i] = 1


    d1 = (ring_dip.y - ring_pip.y)**2 + (ring_dip.x - ring_pip.x)**2
    d2 = (index_tip.y - middle_tip.y)**2 + (index_tip.x - middle_tip.x)**2

    if d2 < d1:
        for i in range(9):
            correct[i] = 0

    a = 0
    for i in range(21):
        a = a + correct[i]
        
    accuracy = (a / 21) * 100
    
    if a == 21:
        return "V", accuracy, correct
    else:
        return "Not V", accuracy, correct

# Define a function to draw landmarks and connections on the image
def draw_landmarks(image, landmarks, correct):
    connections = mp_hands.HAND_CONNECTIONS
    h, w, _ = image.shape

    for idx, landmark in enumerate(landmarks):
        color = (0, 255, 0) if correct[idx] == 1 else (0, 0, 255)
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, color, -1)
    
    for connection in connections:
        start_idx, end_idx = connection
        # Retrieve the start and end points of the connection
        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]
        
        # Determine the color based on correctness
        color = (0, 255, 0) if correct[start_idx] == 1 and correct[end_idx] == 1 else (0, 0, 255)
        
        # Convert landmarks' coordinates to pixel values
        start = (int(start_point.x * w), int(start_point.y * h))
        end = (int(end_point.x * w), int(end_point.y * h))
        
        # Draw a line between the landmarks
        cv2.line(image, start, end, color, 2)

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
    
    # Draw the hand annotations on the image and classify the 'U' gesture
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Calculate hand angles
            h, w, _ = image.shape
            index_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_finger_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            middle_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            
            angle_2d = calculate_2d_angle(index_finger_dip, index_finger_mcp, middle_finger_dip, middle_finger_mcp, h, w)
            angle_2d_1 = calculate_2d_angle(index_finger_tip, index_finger_mcp, wrist, index_finger_mcp, h, w)
            angle_2d_2 = calculate_2d_angle(thumb_tip, thumb_ip, thumb_mcp, thumb_ip, h, w)
            angle_2d_3 = calculate_2d_angle(thumb_tip, wrist, index_finger_tip, wrist, h, w)
            angle_2d_4 = calculate_2d_angle(index_finger_tip, index_finger_pip, index_finger_mcp, index_finger_pip, h, w)
            
            # Classify gesture
            gesture, accuracy, correct = classify_v_gesture(hand_landmarks.landmark, angle_2d_1, angle_2d_2, angle_2d_3, angle_2d_4)
            
            # Draw landmarks and connections with colors based on correctness
            draw_landmarks(image, hand_landmarks.landmark, correct)
            
            # Display  gesture, and accuracy
            cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f'Accuracy: {accuracy:.2f}%', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    
    # Display the image
    cv2.imshow('Hand Tracking', image)
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
