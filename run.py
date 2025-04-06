import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = f.readlines()
    return [label.strip() for label in labels]
 
model = load_model('D:/signmodel/model_hand.h5')

labels = load_labels('D:/signmodel/labels.txt') 

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False  # Performance optimization
    results = model.process(image)  # Make detection
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])
sequence_length = 30
sequence = []  


last_predicted_label = "Waiting for prediction..."

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from camera.")
    exit()

start_time = time.time()


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_time = time.time() - start_time

        if elapsed_time >= 6:
        
            image, results = mediapipe_detection(frame, holistic)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)  
            if len(sequence) == sequence_length:
                
                input_data = np.expand_dims(sequence, axis=0)

                predictions = model.predict(input_data)

                predicted_index = np.argmax(predictions)

               
                last_predicted_label = labels[predicted_index]

                sequence = []


            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, last_predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        cv2.imshow('Camera Feed', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
