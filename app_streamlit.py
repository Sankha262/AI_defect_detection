import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import cv2
import numpy as np
import os

st.set_page_config(page_title="AI Defect Detection", layout="wide")

st.title("🧠 AI-Based Defect Detection System")
st.markdown("Upload an image or use a sample to detect defects using your trained YOLO model.")

@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "best.pt")
    model = YOLO(model_path)
    return model

model = load_model()

# ------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # ------------------------------------------------------
    # RUN YOLO DETECTION
    # ------------------------------------------------------
    st.write("🔍 Running detection...")
    results = model.predict(source=temp_path, conf=0.25)

    # Get result image (first prediction)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption="✅ Detection Result", use_column_width=True)

    # ------------------------------------------------------
    # SHOW DETECTION DETAILS
    # ------------------------------------------------------
    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        st.subheader("📊 Detection Summary")
        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls] if cls in model.names else "Unknown"
            st.write(f"**Object {i+1}:** {label} (Confidence: {conf:.2f})")
    else:
        st.warning("No objects detected.")
else:
    st.info("Please upload an image to start detection.")

# ------------------------------------------------------
# OPTIONAL: SAMPLE IMAGE BUTTON
# ------------------------------------------------------
if st.button("Use Sample Image"):
    st.info("Please add a sample image in your repo if you want to enable this feature.")
