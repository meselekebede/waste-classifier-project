# app.py

import streamlit as st
import torch
from PIL import Image
import predict_waste

# Set page config
st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #2c3e50;
    }
    .prediction-box {
        padding: 15px;
        margin-top: 10px;
        border-radius: 10px;
        font-size: 1.2em;
        font-weight: bold;
        text-align: center;
    }
    .cardboard { background-color: #d0ecf2; color: #1f3a5f; }
    .glass { background-color: #f0f5d8; color: #3b5d20; }
    .metal { background-color: #e2e4e6; color: #2f2f2f; }
    .paper { background-color: #fff9c4; color: #4a4a24; }
    .plastic { background-color: #ffe0b2; color: #663d00; }
    .trash { background-color: #ffcdd2; color: #6e1a1f; }
</style>
""", unsafe_allow_html=True)

# Title & Description
st.title("♻️ Waste Classification")
st.markdown("Upload an image and our AI will classify the type of waste.")

@st.cache_resource
def load_the_model():
    return predict_waste.load_model()

model = load_the_model()

# Upload Image
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess and Predict
    image_tensor = predict_waste.preprocess_image(image)
    prediction, confidence = predict_waste.predict_with_confidence(model, image_tensor)

    # Show Result
    st.markdown(f"<div class='prediction-box {prediction.lower()}'>{prediction}</div>", unsafe_allow_html=True)

    # Confidence bar chart
    st.markdown("### 🔍 Confidence Scores")
    st.bar_chart(confidence)