import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image
import pandas as pd
import cv2
import tempfile
import os

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Defective Product Detection", layout="wide")

@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)

def run_inference(model, img):
    results = model.predict(source=np.array(img), save=False, imgsz=640, conf=0.25)
    annotated = results[0].plot()  # Annotated image
    detections = results[0]
    return annotated, detections

# ----------------- APP START -----------------
st.title("🧠 Defective Product Detection System")
st.markdown("### Upload an image to detect and analyze manufacturing defects.")

weights_path = "runs/defect_yolov12n/weights/best.pt"
model = load_model(weights_path)

# Sidebar
st.sidebar.header("Settings")
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
iou_thres = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("---")

    if st.button("🔍 Detect Defects"):
        with st.spinner("Detecting defects..."):
            annotated, detections = run_inference(model, image)
            st.image(annotated, caption="Detection Results", use_column_width=True)

            # Detection summary
            boxes = detections.boxes
            names = model.names
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            if len(boxes) > 0:
                st.subheader("📋 Detection Summary")
                data = []
                for i in range(len(boxes)):
                    cls_name = names[int(classes[i])]
                    conf = float(confs[i])
                    data.append([i+1, cls_name, f"{conf:.2f}"])
                df = pd.DataFrame(data, columns=["#", "Class", "Confidence"])
                st.dataframe(df, use_container_width=True)

                # Count per class
                st.subheader("📊 Class-wise Count")
                class_counts = pd.Series([names[int(c)] for c in classes]).value_counts()
                st.bar_chart(class_counts)

            else:
                st.warning("No defects detected.")

            # Model performance metrics
            try:
                st.subheader("⚙️ Model Precision Metrics")
                metrics = model.val(data="data/data.yaml")
                st.write(f"**mAP50:** {metrics.box.map50:.4f}")
                st.write(f"**mAP50-95:** {metrics.box.map:.4f}")
                st.write(f"**Precision:** {metrics.box.mp:.4f}")
                st.write(f"**Recall:** {metrics.box.mr:.4f}")
            except Exception as e:
                st.info("Run full validation separately for precise metrics (requires dataset).")

else:
    st.info("👆 Upload an image file to get started.")
