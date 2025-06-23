import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("rice_brownspot_model.keras")

st.title("🌾 Rice Leaf Disease Detector")
uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img_array = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    st.image(img, caption="Uploaded image", use_column_width=True)

    img_resized = cv2.resize(img, (128, 128)) / 255.0
    prediction = model.predict(np.expand_dims(img_resized, axis=0))[0][0]

    label = "Brown Spot" if prediction > 0.5 else "Healthy"
    st.success(f"Prediction: **{label}** ({prediction:.2f})")
