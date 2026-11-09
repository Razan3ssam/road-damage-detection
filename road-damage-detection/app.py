import io
from collections import Counter
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage AI", page_icon="🛣️", layout="wide",
                   initial_sidebar_state="collapsed")

BG, CARD, PRIMARY, ACCENT, TEXT, BORDER = "#0E0B1D", "#2A1E4D", "#5A3B8C", "#8B7CC7", "#C9C2E3", "#3A3A52"

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
section[data-testid="stSidebar"], div[data-testid="stSidebarCollapsedControl"],
div[data-testid="collapsedControl"] {display: none;}
.stApp {background: radial-gradient(circle at 20% 0%, __CARD__ 0%, __BG__ 45%);}
.block-container {padding: 1rem 3rem 3rem 3rem; max-width: 100%;}
html, body, [class*="css"], p, label, span {color: __TEXT__;}

.navbar {display:flex; justify-content:space-between; align-items:center;
  padding: 14px 22px; border:1px solid __BORDER__; border-radius:14px;
  background: rgba(42,30,77,.45); margin-bottom: 24px;}
.brand {font-weight:800; font-size:1.05rem; color:#fff; letter-spacing:.5px;}
.brand span {color:__ACCENT__;}
.navlinks {display:flex; gap:26px; font-size:.85rem; color:__TEXT__; opacity:.85;}
.navbtn {background:__PRIMARY__; color:#fff; padding:7px 18px; border-radius:8px; font-size:.82rem; font-weight:600;}

.badge {display:inline-block; padding:5px 12px; border:1px solid __PRIMARY__; border-radius:999px;
  background: rgba(90,59,140,.25); color:__ACCENT__; font-size:.72rem; letter-spacing:.8px; margin-bottom:18px;}
.hero-title {font-size:3.1rem; font-weight:800; line-height:1.1; color:#fff; margin:0 0 6px 0;}
.hero-title .acc {color:__ACCENT__;}
.hero-sub {font-size:.98rem; line-height:1.6; opacity:.85; margin: 14px 0 26px 0; max-width:520px;}

.panel-title {font-size:.75rem; letter-spacing:1px; color:__ACCENT__; margin-bottom:10px;}
.chip {display:inline-block; background:__PRIMARY__; color:#fff; border-radius:8px;
  padding:6px 12px; margin:4px 6px 4px 0; font-size:.82rem; font-weight:600;}
.section-title {font-size:1.5rem; font-weight:800; color:#fff; margin: 34px 0 12px 0;}

div[data-testid="stFileUploader"] section {background:__BG__; border:1.5px dashed __PRIMARY__; border-radius:12px;}
div[data-testid="stFileUploader"] button, .stDownloadButton button {
  background:__PRIMARY__; color:#fff; border:none; border-radius:8px; font-weight:600;}
.stDownloadButton button:hover, div[data-testid="stFileUploader"] button:hover {background:__ACCENT__; color:__BG__;}
div[data-testid="stMetric"] {background: rgba(42,30,77,.7); border:1px solid __BORDER__; border-radius:12px; padding:14px;}
div[data-testid="stMetricValue"] {color:__ACCENT__;}
</style>
"""
for key, val in {"__BG__": BG, "__CARD__": CARD, "__PRIMARY__": PRIMARY,
                 "__ACCENT__": ACCENT, "__TEXT__": TEXT, "__BORDER__": BORDER}.items():
    CSS = CSS.replace(key, val)
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("v11n_best.pt")


model = load_model()

# ---------- Navbar ----------
st.markdown("""
<div class="navbar">
  <div class="brand">🛣️ ROAD<span>DAMAGE</span> AI</div>
  <div class="navlinks"><span>Home</span><span>Detection</span><span>Results</span><span>About</span></div>
  <div class="navbtn">YOLO11n</div>
</div>
""", unsafe_allow_html=True)

# Settings
s1, s2, _ = st.columns([1, 1, 2])
conf = s1.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
iou = s2.slider("IoU threshold (NMS)", 0.10, 0.90, 0.70, 0.05)

# Hero
left, right = st.columns([1, 1.1], gap="large")

with left:
    st.markdown("""
    <div class="badge">◈ ROAD DAMAGE AI — INFRASTRUCTURE INTELLIGENCE</div>
    <div class="hero-title">See the Damage.<br><span class="acc">Detect the Type.</span><br>Fix Roads Smarter.</div>
    <div class="hero-sub">AI-powered road damage detection using YOLO11n. Upload a road photo and get
    bounding boxes, damage types, and confidence scores for 7 damage classes.</div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="panel-title">● LIVE AI INFERENCE</div>', unsafe_allow_html=True)
    file = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

    result, annotated, image = None, None, None
    if file:
        image = Image.open(file).convert("RGB")
        result = model.predict(np.array(image), conf=conf, iou=iou, imgsz=640, verbose=False)[0]
        annotated = result.plot()[:, :, ::-1]
        st.image(annotated, width="stretch")
    else:
        st.info("Upload a road image (jpg / png) to start detection.")

# Results
if result is not None:
    names = result.names
    counts = Counter(names[int(c)] for c in result.boxes.cls)
    confs = [float(b.conf) for b in result.boxes]

    st.markdown('<div class="section-title">Detection Results</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("Total damages", len(result.boxes))
    m2.metric("Damage types", len(counts))
    m3.metric("Avg confidence", f"{(sum(confs) / len(confs)):.2f}" if confs else "-")

    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.markdown('<div class="panel-title">DETECTED CLASSES</div>', unsafe_allow_html=True)
        if counts:
            chips = "".join(f'<span class="chip">{k} x {n}</span>' for k, n in counts.items())
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.write("No damage detected. Try lowering the confidence threshold.")
        st.markdown('<div class="panel-title" style="margin-top:20px">ORIGINAL IMAGE</div>',
                    unsafe_allow_html=True)
        st.image(image, width="stretch")

    with c2:
        if counts:
            rows = [{
                "Class": names[int(b.cls)],
                "Confidence": round(float(b.conf), 3),
                "Box (x1,y1,x2,y2)": str([int(v) for v in b.xyxy[0].tolist()]),
            } for b in result.boxes]
            st.dataframe(rows, width="stretch", height=330)

        buf = io.BytesIO()
        Image.fromarray(annotated).save(buf, format="PNG")
        st.download_button("Download result image", buf.getvalue(),
                           file_name="detection.png", mime="image/png")