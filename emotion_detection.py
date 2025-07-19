# emotion_detection.py

import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

def emotion_detection():
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    model = load_model('emotion_model.h5')  # Load trained model

    cap = cv2.VideoCapture(0)
    haarcascade_path = "haarcascade_frontalface_default.xml"
    
    if not os.path.exists(haarcascade_path):
        raise ValueError(f"Cannot find {haarcascade_path}. Please download and place it in the same directory.")

    face_cascade = cv2.CascadeClassifier(haarcascade_path)
    if face_cascade.empty():
        raise ValueError("Error loading Haarcascade. File may be corrupted.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.reshape(1, 48, 48, 1) / 255.0
            prediction = model.predict(roi)
            emotion = emotion_labels[np.argmax(prediction)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Real-Time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_detection()
