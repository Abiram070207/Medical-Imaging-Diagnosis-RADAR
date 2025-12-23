import os
import urllib.request
import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from gradcam import generate_gradcam, overlay_heatmap

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical Imaging Diagnosis",
    layout="wide"
)

# -----------------------------
# HEADER WITH LOGO
# -----------------------------
logo = Image.open("radar_logo.png")

col1, col2 = st.columns([7, 3])

with col1:
    st.title("🩺 AI-Based Medical Imaging Diagnosis")
    st.subheader("Chest X-ray Pneumonia Detection (RADAR Project)")

with col2:
    st.image(logo, width=180)

st.markdown(
    """
    This system assists radiologists by automatically screening chest X-ray images
    for pneumonia and highlighting suspicious regions using explainable AI (Grad-CAM).
    The system is designed as a **decision-support tool**, not a replacement for clinicians.
    """
)

st.markdown("---")

# -----------------------------
# MODEL CONFIG (HUGGING FACE)
# -----------------------------
MODEL_URL = (
    "https://huggingface.co/abiram07/pneumonia-xray-radar/"
    "resolve/main/pneumonia_model.h5"
)
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "pneumonia_model.h5")

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading AI model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    return load_model(MODEL_PATH)

try:
    model = load_trained_model()
except Exception as e:
    st.error("⚠️ Unable to load AI model. Please try again later.")
    st.stop()

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(img, img_size=224):
    img = cv2.resize(img, (img_size, img_size))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# DECISION THRESHOLDS
# -----------------------------
PNEUMONIA_THRESHOLD = 0.70
NORMAL_THRESHOLD = 0.40

# -----------------------------
# PREDICTION & VISUALIZATION
# -----------------------------
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    # Display uploaded image
    with col1:
        st.image(
            img,
            caption="🩻 Uploaded Chest X-ray",
            use_container_width=True
        )

    # Prediction
    img_input = preprocess_image(img)
    prediction = model.predict(img_input)[0][0]

    st.markdown("---")
    st.subheader("🧠 AI Prediction")

    if prediction >= PNEUMONIA_THRESHOLD:
        label = "🟥 Pneumonia Detected"
        confidence = prediction * 100
        st.error(f"**Result:** {label}")

    elif prediction <= NORMAL_THRESHOLD:
        label = "🟩 Normal"
        confidence = (1 - prediction) * 100
        st.success(f"**Result:** {label}")

    else:
        label = "🟨 Inconclusive – Needs Radiologist Review"
        confidence = 100 - abs(50 - prediction * 100)
        st.warning(f"**Result:** {label}")

    st.write(f"**Confidence:** {confidence:.2f}%")
    st.progress(int(confidence))

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    st.markdown("---")
    st.subheader("🔥 Explainable AI – Grad-CAM")

    temp_path = "temp_xray.jpg"
    cv2.imwrite(temp_path, img)

    heatmap = generate_gradcam(model, temp_path)
    cam_image = overlay_heatmap(temp_path, heatmap)

    _, buffer = cv2.imencode(".jpg", cam_image)
    heatmap_bytes = buffer.tobytes()

    st.download_button(
        label="⬇️ Download Grad-CAM Heatmap",
        data=heatmap_bytes,
        file_name="gradcam_heatmap.jpg",
        mime="image/jpeg"
    )

    with col2:
        st.image(
            cam_image,
            caption="Grad-CAM Heatmap (Highlighted Regions)",
            use_container_width=True
        )

    st.info(
        "Highlighted regions represent areas that influenced the AI model's decision."
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption(
    "RADAR – Rajalakshmi Advanced Diagnostics and Research | AI & Data Science Project"
)
