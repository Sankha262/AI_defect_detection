import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image
import os
import numpy as np

# ------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(page_title="AI Defect Detection", layout="wide")
st.title("🧠 AI-Based Defect Detection System")
st.markdown("Upload an image to detect defects using your trained YOLO model.")

# ------------------------------------------------------
# LOAD MODEL SAFELY
# ------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.getcwd(), "best.pt")

    # Debug info for file presence
    st.write("📁 Model path:", model_path)
    exists = os.path.exists(model_path)
    st.write("✅ Exists:", exists)

    if not exists:
        st.error("❌ Model file not found. Please make sure 'best.pt' is in the same folder.")
        st.stop()

    size = os.path.getsize(model_path)
    st.write(f"📏 Model size: {size/1e6:.2f} MB")

    if size < 1e5:
        st.error("❌ 'best.pt' seems too small — it may not be a valid YOLO weight file.")
        st.stop()

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

model = load_model()

# ------------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_path = temp_file.name

    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # ------------------------------------------------------
    # RUN YOLO PREDICTION
    # ------------------------------------------------------
    st.write("🔍 Running detection...")
    try:
        results = model.predict(source=temp_path, conf=0.25)
        res_plotted = results[0].plot()  # numpy image
        st.image(res_plotted, caption="✅ Detection Result", use_column_width=True)

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
    except Exception as e:
        st.error(f"❌ Error during detection: {e}")
else:
    st.info("Please upload an image to start detection.")
