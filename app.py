import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="PCOS Region Detector", page_icon="🎀", layout="centered")

st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #fff0f5 0%, #ffe4ec 100%); }
    .main { padding-top: 1rem; }
    h1 {
        color: #d6336c;
        font-weight: 800;
        text-align: center;
        font-family: 'Comic Sans MS', 'Trebuchet MS', sans-serif;
    }
    .subtitle {
        color: #a64d79;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 1.5rem;
        font-family: 'Trebuchet MS', sans-serif;
    }
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #ff8fab;
        border-radius: 18px;
        padding: 12px;
    }
    .metric-box {
        background: #ffffff;
        border: 2px solid #ffb3c6;
        border-radius: 20px;
        padding: 18px 22px;
        margin-top: 18px;
        box-shadow: 0 4px 12px rgba(255, 143, 171, 0.25);
    }
    .metric-row { display: flex; justify-content: space-around; text-align: center; }
    .metric-value { font-size: 1.9rem; font-weight: 800; font-family: 'Trebuchet MS', sans-serif; }
    .metric-label {
        font-size: 0.78rem;
        color: #c9184a;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .infected { color: #e5383b; }
    .notinfected { color: #52b788; }
    .total { color: #d6336c; }
    .stAlert { border-radius: 16px; }
    img { border-radius: 18px; border: 3px solid #ffb3c6; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎀 PCOS Ultrasound Region Detector 🎀</h1>", unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">A cute lil YOLOv8 model spotting infected vs. not-infected ovarian regions ✨<br>'
    'Research/portfolio demo only — not a diagnostic tool 💕</p>',
    unsafe_allow_html=True
)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded = st.file_uploader("💮 Upload an ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Sprinkling some detection magic... ✨"):
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
                    <div class="metric-label">🎀 Infected</div>
                </div>
                <div>
                    <div class="metric-value notinfected">{notinfected}</div>
                    <div class="metric-label">🎀 Not Infected</div>
                </div>
                <div>
                    <div class="metric-value total">{total}</div>
                    <div class="metric-label">🎀 Total</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.info("💗 Upload an ultrasound frame to see the detections!")
