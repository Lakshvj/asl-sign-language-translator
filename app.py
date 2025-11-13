import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("asl_model.h5")
with open("label_names.json", "r") as f:
    label_names = json.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

# Extract 21 landmarks (63 values)
def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if not result.multi_hand_landmarks:
        return None
    
    hand = result.multi_hand_landmarks[0]
    landmarks = []
    for lm in hand.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
    
    return np.array(landmarks).reshape(1, -1)

# Streamlit UI
st.title("✋ AI-Based Sign Language Translator")
st.write("Show a sign in front of your camera to get prediction.")

run = st.checkbox("Start Camera")

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera not detected")
        break

    landmarks = extract_landmarks(frame)
    
    if landmarks is not None:
        pred = model.predict(landmarks)[0]
        idx = np.argmax(pred)
        sign = label_names[idx]
        confidence = pred[idx]

        cv2.putText(frame, f"{sign} ({confidence:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
camera.release()
