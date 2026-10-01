import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="PCOS Region Detector", page_icon="🩺", layout="centered")

st.markdown("""
    <style>
    .main { padding-top: 1rem; }
    .stApp { background-color: #0e1117; }
    h1 { color: #e0e0e0; font-weight: 700; }
    .subtitle { color: #9ca3af; font-size: 0.95rem; margin-bottom: 1.5rem; }
    .metric-box {
        background: #1a1d24;
        border: 1px solid #2d3139;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 16px;
    }
    .metric-row { display: flex; justify-content: space-around; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
    .infected { color: #f87171; }
    .notinfected { color: #4ade80; }
    .total { color: #60a5fa; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🩺 PCOS Ultrasound Region Detector</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">YOLOv8 model detecting infected vs. not-infected ovarian regions. '
    'Research/portfolio demo only — not a diagnostic tool.</p>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("Upload an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Running detection..."):
        results = model.predict(np.array(image), verbose=False)
    r = results[0]
    annotated = r.plot()

    infected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "infected")
    notinfected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "notinfected")
    total = len(r.boxes)

    st.image(annotated, channels="BGR", use_container_width=True)

    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-row">
                <div>
                    <div class="metric-value infected">{infected}</div>
                    <div class="metric-label">Infected</div>
                </div>
                <div>
                    <div class="metric-value notinfected">{notinfected}</div>
                    <div class="metric-label">Not Infected</div>
                </div>
                <div>
                    <div class="metric-value total">{total}</div>
                    <div class="metric-label">Total</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("Upload an ultrasound frame to see detections.")
