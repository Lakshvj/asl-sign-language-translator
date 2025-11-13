import streamlit as st
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import json

# Load your trained model
import tensorflow as tf
model = tf.keras.models.load_model("asl_model.h5")

# Load label mapping
with open("label_names.json", "r") as f:
    label_names = json.load(f)

detector = HandDetector(maxHands=1)

st.title("ASL Sign Language Translator")

run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    hands, img = detector.findHands(frame)

    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # 21 hand landmarks

        # Flatten to 63 inputs for your model
        data = np.array(lmList).flatten().reshape(1, 63)

        pred = model.predict(data)[0]
        idx = np.argmax(pred)
        confidence = pred[idx]

        prediction = f"{label_names[idx]} ({confidence:.2f})"
        cv2.putText(img, prediction, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    FRAME_WINDOW.image(img, channels="BGR")

cap.release()
