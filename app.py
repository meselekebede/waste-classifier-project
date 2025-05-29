import streamlit as st
from PIL import Image
import io
import predict_waste

# Page config
st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")

# Custom CSS styling
st.markdown("""
<style>
    body {
        background-color: #f8f9fa;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-header {
        color: #2c3e50;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-header {
        font-size: 1.3em;
        text-align: center;
        margin-bottom: 20px;
        color: #555;
    }
    .instruction-box {
        background-color: #e8f4e3;
        border-left: 6px solid #27ae60;
        padding: 15px;
        margin: 10px 0;
        font-size: 1.1em;
        color: #2d5d2c;
    }
    .supported-classes {
        background-color: #f0f0f0;
        border-radius: 10px;
        padding: 15px;
        margin-top: 10px;
        font-size: 1em;
        color: #333;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 0.9em;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">♻️ AI Waste Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload an image of waste, and our ResNet-34 model will classify it into one of six categories.</div>', unsafe_allow_html=True)

# Project Description
with st.expander("ℹ️ About This App"):
    st.markdown("""
This model is on ealy stage with the accuracy of 83% and can make unexpected errors, this just for fun, don't take it for real.
    This app uses a deep learning model based on **ResNet-34 architecture** trained on real-world waste images.
    
    It supports classification into the following categories:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash
    
    The model was trained using transfer learning from PyTorch's pre-trained weights and fine-tuned only the final layer.
    
    🚀 Deployed with ❤️ using Streamlit.
    """)

# Instruction Box
st.markdown('<div class="instruction-box">📷 <strong>Click "Choose an image..." below</strong> to upload a photo of waste (JPG, JPEG, or PNG format).</div>', unsafe_allow_html=True)

# Supported Classes
st.markdown('<div class="instruction-box">🔍 After uploading, the model will analyze the image and show you the predicted waste type along with confidence scores.</div>', unsafe_allow_html=True)

st.markdown('<div class="supported-classes"><strong>Supported Waste Types:</strong><ul>'
            '<li>📦 Cardboard</li>'
            '<li>🪟 Glass</li>'
            '<li>🔩 Metal</li>'
            '<li>📄 Paper</li>'
            '<li>🥤 Plastic</li>'
            '<li>🗑️ Trash</li>'
            '</ul></div>', unsafe_allow_html=True)

# Load model once
@st.cache_resource
def load_the_model():
    return predict_waste.load_model()

model = load_the_model()

# Upload Image
uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Read image
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Display image
        st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)

        # Predict
        image_tensor = predict_waste.preprocess_image(image)
        prediction, confidence = predict_waste.predict_with_confidence(model, image_tensor)

        # Show Result
        st.markdown(f"<h3 style='text-align:center; color:green;'>🧾 Prediction: {prediction.capitalize()}</h3>", unsafe_allow_html=True)

        # Confidence bar chart
        st.markdown("### 🔍 Confidence Scores")
        st.bar_chart(confidence)

    except Exception as e:
        st.error(f"❌ An error occurred while processing the image: {e}")
        st.warning("Please try uploading a different image.")
else:
    st.markdown('<p style="color:#999; text-align:center;">No image uploaded yet.</p>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Built with ❤️ using PyTorch & Streamlit</div>', unsafe_allow_html=True)