import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.title("PCOS Ultrasound Region Detector")
st.write("YOLOv8 model detecting infected vs. not-infected ovarian regions. Research/portfolio demo only — not a diagnostic tool.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    results = model.predict(np.array(image))
    r = results[0]
    annotated = r.plot()

    infected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "infected")
    notinfected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "notinfected")

    st.image(annotated, caption="Detections", channels="BGR")
    st.write(f"**Infected:** {infected} | **Not infected:** {notinfected} | **Total detections:** {len(r.boxes)}")
