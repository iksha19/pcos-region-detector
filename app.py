import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="PCOS Ultrasound Region Detector", page_icon="🎀", layout="centered")

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap');

:root {
    --blush: #F8C9D9;
    --strawberry: #F3AFC8;
    --lavender: #DCD3F3;
    --periwinkle: #C9D5F4;
    --gold: #EBCB79;
    --plum: #694A73;
    --ivory: #FFFCF5;
}

html, body, [class*="css"]  {
    font-family: 'Nunito', sans-serif;
}

.stApp {
    background: linear-gradient(160deg, #FFF5F8 0%, #FBEFF6 35%, #F3EEFB 65%, #FFFCF5 100%);
    background-attachment: fixed;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    opacity: 0.35;
    background-image:
        radial-gradient(circle at 8% 15%, var(--gold) 0px, transparent 2px),
        radial-gradient(circle at 92% 20%, var(--lavender) 0px, transparent 2px),
        radial-gradient(circle at 15% 80%, var(--strawberry) 0px, transparent 2px),
        radial-gradient(circle at 85% 75%, var(--periwinkle) 0px, transparent 2px),
        radial-gradient(circle at 50% 92%, var(--gold) 0px, transparent 2px),
        radial-gradient(circle at 70% 8%, var(--blush) 0px, transparent 2px);
    background-size: 100% 100%;
    z-index: 0;
}

.eyebrow {
    text-align: center;
    color: var(--plum);
    opacity: 0.65;
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

.sun-ornament {
    display: flex;
    justify-content: center;
    margin-bottom: 0.2rem;
}
.sun-ornament svg { animation: glow 3.5s ease-in-out infinite; }
@keyframes glow {
    0%, 100% { filter: drop-shadow(0 0 2px rgba(235,203,121,0.4)); }
    50% { filter: drop-shadow(0 0 8px rgba(235,203,121,0.8)); }
}
@media (prefers-reduced-motion: reduce) {
    .sun-ornament svg { animation: none; }
}

h1.hero-title {
    font-family: 'Quicksand', sans-serif;
    font-weight: 700;
    text-align: center;
    color: var(--plum);
    font-size: 2.1rem;
    margin: 0.2rem 0 0.3rem 0;
}

.hero-subtitle {
    text-align: center;
    color: var(--plum);
    opacity: 0.75;
    font-style: italic;
    font-size: 1rem;
    margin-bottom: 0.4rem;
}

.hero-disclaimer {
    text-align: center;
    color: var(--plum);
    opacity: 0.55;
    font-size: 0.78rem;
    margin-bottom: 1.6rem;
}

.divider-flowers {
    text-align: center;
    opacity: 0.5;
    margin: 0.6rem 0 1.4rem 0;
    font-size: 0.9rem;
    color: var(--strawberry);
}

.upload-heading {
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    color: var(--plum);
    font-size: 1.15rem;
    margin-bottom: 0.2rem;
    text-align: center;
}
.upload-sub {
    text-align: center;
    color: var(--plum);
    opacity: 0.6;
    font-size: 0.85rem;
    margin-bottom: 0.8rem;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.55) !important;
    border: 2px dashed var(--strawberry) !important;
    border-radius: 26px !important;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    box-shadow: 0 0 18px rgba(243,175,200,0.45);
}

.canvas-label {
    text-align: center;
    font-family: 'Quicksand', sans-serif;
    font-weight: 600;
    color: var(--plum);
    opacity: 0.7;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.5rem 0;
}

.canvas-frame {
    background: var(--ivory);
    border-radius: 24px;
    padding: 14px;
    box-shadow: 0 8px 24px rgba(220, 211, 243, 0.45);
    border: 1px solid rgba(220,211,243,0.6);
    position: relative;
    animation: fadein 0.6s ease;
}
@keyframes fadein {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.canvas-frame img {
    border-radius: 16px;
    width: 100%;
    display: block;
}

.stats-row {
    display: flex;
    gap: 14px;
    margin-top: 1.4rem;
}
.stat-card {
    flex: 1;
    border-radius: 20px;
    padding: 18px 10px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(105,74,115,0.1);
    border: 1px solid rgba(255,255,255,0.6);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 22px rgba(105,74,115,0.18);
}
.stat-card.rose { background: linear-gradient(160deg, #FDEFF4, #FBE1EC); }
.stat-card.lav  { background: linear-gradient(160deg, #F2EEFB, #E7E0F8); }
.stat-card.cream{ background: linear-gradient(160deg, #FDF8EA, #F7EAC9); }

.stat-emoji { font-size: 1.1rem; margin-bottom: 2px; }
.stat-value {
    font-family: 'Quicksand', sans-serif;
    font-weight: 700;
    font-size: 2rem;
    color: var(--plum);
}
.stat-label {
    font-family: 'Nunito', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--plum);
    opacity: 0.6;
    margin-top: 2px;
}

.footer-note {
    text-align: center;
    margin-top: 2.2rem;
    color: var(--plum);
    opacity: 0.45;
    font-size: 0.75rem;
}

@media (max-width: 600px) {
    .stats-row { flex-direction: column; }
    h1.hero-title { font-size: 1.6rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">✧ A LITTLE LABORATORY OF AI MAGIC ✧</div>', unsafe_allow_html=True)

st.markdown("""
<div class="sun-ornament">
<svg width="46" height="46" viewBox="0 0 46 46" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="23" cy="23" r="9" fill="#EBCB79"/>
  <g stroke="#EBCB79" stroke-width="2" stroke-linecap="round">
    <line x1="23" y1="2" x2="23" y2="8"/>
    <line x1="23" y1="38" x2="23" y2="44"/>
    <line x1="2" y1="23" x2="8" y2="23"/>
    <line x1="38" y1="23" x2="44" y2="23"/>
    <line x1="8.5" y1="8.5" x2="12.5" y2="12.5"/>
    <line x1="33.5" y1="33.5" x2="37.5" y2="37.5"/>
    <line x1="8.5" y1="37.5" x2="12.5" y2="33.5"/>
    <line x1="33.5" y1="12.5" x2="37.5" y2="8.5"/>
  </g>
</svg>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">🎀 PCOS Ultrasound Region Detector</h1>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Where a little science meets a little magic ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-disclaimer">Research &amp; portfolio demonstration only — not a diagnostic tool.</div>', unsafe_allow_html=True)
st.markdown('<div class="divider-flowers">✿ ·｡ ✧ ｡· ✿</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.markdown('<div class="upload-heading">Let\'s begin our little discovery</div>', unsafe_allow_html=True)
st.markdown('<div class="upload-sub">Drop your image or browse your files ✨</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Place your ultrasound image here",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) > 10:
    st.warning("✦ Please upload up to 10 images at a time — only the first 10 will be analyzed ✦")
    uploaded_files = uploaded_files[:10]

if uploaded_files:
    total_infected = 0
    total_notinfected = 0

    for idx, uploaded in enumerate(uploaded_files):
        image = Image.open(uploaded).convert("RGB")
        with st.spinner(f"Sprinkling a little detection magic on image {idx+1}/{len(uploaded_files)}... ✨"):
            results = model.predict(np.array(image), verbose=False)
        r = results[0]
        annotated = r.plot()

        infected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "infected")
        notinfected = sum(1 for b in r.boxes if model.names[int(b.cls)] == "notinfected")
        total = len(r.boxes)
        total_infected += infected
        total_notinfected += notinfected

        st.markdown(f'<div class="canvas-label">✦ Analysis Canvas — Image {idx+1} of {len(uploaded_files)} ✦</div>', unsafe_allow_html=True)

        annotated_rgb = annotated[:, :, ::-1]
        from io import BytesIO
        import base64
        buf = BytesIO()
        Image.fromarray(annotated_rgb).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        st.markdown(f"""
            <div class="canvas-frame">
                <img src="data:image/png;base64,{b64}" />
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card rose">
                    <div class="stat-emoji">🎀</div>
                    <div class="stat-value">{infected}</div>
                    <div class="stat-label">Infected</div>
                </div>
                <div class="stat-card lav">
                    <div class="stat-emoji">🌷</div>
                    <div class="stat-value">{notinfected}</div>
                    <div class="stat-label">Not Infected</div>
                </div>
                <div class="stat-card cream">
                    <div class="stat-emoji">☀️</div>
                    <div class="stat-value">{total}</div>
                    <div class="stat-label">Total Detections</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="divider-flowers">✿ ·｡ ✧ ｡· ✿</div>', unsafe_allow_html=True)

    if len(uploaded_files) > 1:
        st.markdown(f"""
            <div class="stats-row">
                <div class="stat-card rose">
                    <div class="stat-emoji">🎀</div>
                    <div class="stat-value">{total_infected}</div>
                    <div class="stat-label">Total Infected</div>
                </div>
                <div class="stat-card lav">
                    <div class="stat-emoji">🌷</div>
                    <div class="stat-value">{total_notinfected}</div>
                    <div class="stat-label">Total Not Infected</div>
                </div>
                <div class="stat-card cream">
                    <div class="stat-emoji">📊</div>
                    <div class="stat-value">{len(uploaded_files)}</div>
                    <div class="stat-label">Images Analyzed</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
else:
    st.markdown("""
        <div style="text-align:center; opacity:0.55; color:#694A73; margin-top:1.5rem; font-size:0.9rem;">
        🕯️ Awaiting your ultrasound image(s) to begin the little discovery...
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer-note">
✧ Built as a research &amp; portfolio demonstration · YOLOv8 object detection · not for clinical use ✧
</div>
""", unsafe_allow_html=True)
