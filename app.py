"""
Hampi Revived — Streamlit Web Application

Showcases the full pipeline:
  • Historical context & timeline
  • Raw imagery + preprocessing
  • 3D reconstruction (SfM, dense cloud, mesh)
  • Generative inpainting (SDXL baseline)
  • LoRA-conditioned restoration (ControlNet + rank-8 LoRA)

Run:
    source venv/bin/activate
    streamlit run app.py
"""

import os
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import cv2

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hampi Revived",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT      = Path(__file__).parent
RAW_DIR   = ROOT / "data" / "raw"
TRAIN_DIR = ROOT / "data" / "lora_train"
OUT_DIR   = ROOT / "outputs"
VIZ_DIR   = OUT_DIR / "visualizations"

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=EB+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

  /* ── Global background ── */
  .stApp, .main, [data-testid="stAppViewContainer"] {
      background-color: #0c0a07 !important;
  }
  [data-testid="stHeader"] { background: #0c0a07 !important; }

  /* ── Typography ── */
  body, p, li, .stMarkdown { color: #c8b090 !important; }

  /* ── Hero ── */
  .hero-wrap {
      background: linear-gradient(160deg, #180e04 0%, #261408 40%, #1a1005 80%, #0c0a07 100%);
      border: 1px solid #5a3520;
      border-radius: 14px;
      padding: 3rem 3.5rem 2.5rem;
      margin-bottom: 2rem;
      text-align: center;
      position: relative;
      overflow: hidden;
  }
  .hero-wrap::before {
      content: "𑀓 𑀓 𑀓 𑀓 𑀓";
      position: absolute;
      top: 8px; left: 0; right: 0;
      font-size: 0.7rem;
      color: #3a2010;
      letter-spacing: 12px;
  }
  .hero-title {
      font-family: 'Cinzel', serif;
      font-size: 3.2rem;
      font-weight: 700;
      color: #e8c87a;
      letter-spacing: 6px;
      margin: 0.5rem 0 0.2rem;
      text-shadow: 0 0 40px rgba(200,150,60,0.3);
  }
  .hero-sub {
      font-family: 'Cinzel', serif;
      color: #a07848;
      font-size: 1rem;
      letter-spacing: 3px;
      margin-bottom: 1.2rem;
  }
  .hero-divider {
      border: none;
      border-top: 1px solid #3a2010;
      margin: 1rem auto;
      width: 60%;
  }
  .hero-tagline {
      font-family: 'EB Garamond', serif;
      font-style: italic;
      color: #c09060;
      font-size: 1.25rem;
      line-height: 1.7;
  }
  .hero-badges { margin-top: 1.2rem; }
  .badge {
      display: inline-block;
      background: rgba(90,40,10,0.35);
      border: 1px solid #6a3818;
      border-radius: 4px;
      padding: 0.25rem 0.9rem;
      font-size: 0.72rem;
      color: #c07840;
      margin: 0.2rem 0.3rem;
      letter-spacing: 1px;
  }

  /* ── Section headers ── */
  .sec-hdr {
      font-family: 'Cinzel', serif;
      color: #d4903a;
      font-size: 1.2rem;
      font-weight: 600;
      border-bottom: 1px solid #3a2010;
      padding-bottom: 0.4rem;
      margin: 1.8rem 0 1rem;
      letter-spacing: 2px;
  }
  .sec-sub {
      font-family: 'EB Garamond', serif;
      color: #806040;
      font-size: 0.88rem;
      margin-top: -0.6rem;
      margin-bottom: 1rem;
  }

  /* ── Timeline ── */
  .timeline { position: relative; padding-left: 2rem; }
  .timeline::before {
      content: "";
      position: absolute; left: 0.6rem; top: 0; bottom: 0;
      width: 2px;
      background: linear-gradient(to bottom, #5a3010, #3a1808, #1a0a04);
  }
  .tl-entry { position: relative; margin-bottom: 1.6rem; }
  .tl-dot {
      position: absolute;
      left: -1.85rem; top: 0.35rem;
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #c07840;
      border: 2px solid #5a3010;
      box-shadow: 0 0 8px rgba(192,120,64,0.4);
  }
  .tl-year {
      font-family: 'Cinzel', serif;
      color: #e8a050;
      font-size: 0.82rem;
      font-weight: 600;
      letter-spacing: 1.5px;
  }
  .tl-title {
      color: #d4a870;
      font-weight: 500;
      font-size: 0.95rem;
      margin: 0.1rem 0;
  }
  .tl-body {
      color: #807060;
      font-size: 0.82rem;
      line-height: 1.6;
  }

  /* ── Arch glossary cards ── */
  .arch-card {
      background: #130f0a;
      border: 1px solid #2a1a10;
      border-top: 3px solid #8a4020;
      border-radius: 6px;
      padding: 1rem 1.1rem;
      margin-bottom: 0.6rem;
  }
  .arch-card .term {
      font-family: 'Cinzel', serif;
      color: #d49050;
      font-size: 0.9rem;
      font-weight: 600;
      margin-bottom: 0.3rem;
  }
  .arch-card .def {
      color: #807060;
      font-size: 0.82rem;
      line-height: 1.55;
  }

  /* ── Stage pipeline cards ── */
  .stage-card {
      background: #130f0a;
      border-left: 3px solid #9a4828;
      border-radius: 0 6px 6px 0;
      padding: 0.9rem 1.1rem;
      margin: 0.5rem 0;
  }
  .stage-card h4 { color: #e0a860; margin: 0 0 0.3rem; font-size: 0.92rem; }
  .stage-card p  { color: #706050; margin: 0; font-size: 0.8rem; line-height: 1.5; }

  /* ── Metric cards ── */
  .metric-card {
      background: #130f0a;
      border: 1px solid #2a1808;
      border-radius: 8px;
      padding: 1.1rem 1rem;
      text-align: center;
  }
  .metric-card .val { font-family: 'Cinzel', serif; font-size: 1.7rem; color: #e8c060; }
  .metric-card .lbl { color: #706050; font-size: 0.75rem; margin-top: 0.2rem; }

  /* ── Monument cards ── */
  .monument-card {
      background: linear-gradient(135deg, #130e08, #1a1208);
      border: 1px solid #2a1a10;
      border-radius: 8px;
      padding: 1.2rem;
      margin-bottom: 0.8rem;
  }
  .monument-card .name {
      font-family: 'Cinzel', serif;
      color: #d4a860;
      font-size: 1rem;
      margin-bottom: 0.4rem;
  }
  .monument-card .desc { color: #706050; font-size: 0.82rem; line-height: 1.6; }
  .monument-card .tag  {
      display: inline-block;
      background: #2a1408;
      border: 1px solid #4a2410;
      border-radius: 3px;
      padding: 0.1rem 0.5rem;
      font-size: 0.7rem;
      color: #a06030;
      margin: 0.3rem 0.2rem 0 0;
  }

  /* ── Pill badges ── */
  .pill {
      display: inline-block;
      background: #2a1408;
      border: 1px solid #5a2e10;
      border-radius: 20px;
      padding: 0.12rem 0.65rem;
      font-size: 0.72rem;
      color: #c07840;
      margin: 0.15rem 0.15rem;
  }

  /* ── Before/After containers ── */
  .compare-lbl {
      font-family: 'Cinzel', serif;
      font-size: 0.72rem;
      letter-spacing: 2px;
      color: #806040;
      text-align: center;
      margin-bottom: 0.3rem;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: #0e0b07 !important;
      border-right: 1px solid #2a1a0e !important;
  }
  [data-testid="stSidebar"] p,
  [data-testid="stSidebar"] li { color: #907060 !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
      background: #100d08 !important;
      border-bottom: 1px solid #2a1a10 !important;
  }
  .stTabs [data-baseweb="tab"] { color: #806040 !important; }
  .stTabs [aria-selected="true"] { color: #e8c060 !important; border-bottom: 2px solid #c07040 !important; }

  /* ── Quote block ── */
  .hist-quote {
      font-family: 'EB Garamond', serif;
      font-style: italic;
      font-size: 1.1rem;
      color: #a08060;
      border-left: 3px solid #5a3010;
      padding: 0.8rem 1.2rem;
      margin: 1rem 0;
      background: #110d08;
      border-radius: 0 6px 6px 0;
  }
  .hist-quote .attr { color: #605040; font-size: 0.78rem; margin-top: 0.4rem; }

  /* ── Image overlay label ── */
  .img-label {
      font-size: 0.72rem;
      color: #806040;
      text-align: center;
      font-family: 'Cinzel', serif;
      letter-spacing: 1px;
      padding: 0.2rem 0;
  }

  /* ── Horizontal rule ── */
  hr { border-color: #1e1408 !important; }

  /* ── Caption ── */
  [data-testid="stCaptionContainer"] { color: #605040 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_img(path) -> Image.Image | None:
    p = Path(path)
    return Image.open(p).convert("RGB") if p.exists() else None

def show_img(path, caption="", width=None):
    img = load_img(path)
    if img:
        st.image(img, caption=caption, use_container_width=(width is None))
    else:
        st.markdown(
            f'<div style="background:#110d08;border:1px dashed #2a1808;border-radius:6px;'
            f'padding:2rem;text-align:center;color:#3a2810;font-size:0.8rem;">'
            f'⬜ Not yet generated: <code style="color:#4a3020">{Path(path).name}</code></div>',
            unsafe_allow_html=True
        )

def pill(text: str) -> str:
    return f'<span class="pill">{text}</span>'

def sec(title: str, sub: str = ""):
    st.markdown(f'<div class="sec-hdr">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="sec-sub">{sub}</div>', unsafe_allow_html=True)

def raw_name_map():
    """Map raw image stems to readable monument names."""
    return {
        "083447a317": "Vittala Temple Complex",
        "38580ab157": "Stepped Tank / Pushkarini",
        "59b2b09ec5": "Vijayanagara Gateway Arch",
        "5fd2b1c725": "Matanga Hill Panorama",
        "7f9fc5ee81": "Gopuram — North Entrance",
        "c10d00957e": "Pillared Mandapa",
        "fc23907c75": "Hazara Rama Gopuram",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="font-family:'Cinzel',serif;font-size:1.1rem;color:#e8c060;
letter-spacing:3px;text-align:center;padding:0.5rem 0 1rem">
🏛️ HAMPI REVIVED
</div>
""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
<p style="color:#807060;font-size:0.82rem;line-height:1.7">
<b style="color:#c09050">Vijayanagara Empire</b><br>
1336 – 1646 CE<br><br>
Digital resurrection of one of history's greatest architectural legacies —
using computer vision, 3D reconstruction, and generative AI to restore
monuments abandoned for four centuries.
</p>
""", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="font-family:Cinzel,serif;color:#a07040;font-size:0.78rem;letter-spacing:2px;margin-bottom:0.5rem">PIPELINE</div>', unsafe_allow_html=True)
    stages = [
        ("📥", "Data Ingestion",       "Wikimedia + synthetic"),
        ("🔬", "Preprocessing",        "CLAHE · denoise · sharpen"),
        ("🔍", "Feature Extraction",   "SIFT + FLANN"),
        ("📐", "Structure from Motion","Essential matrix, RANSAC"),
        ("☁️", "Dense Reconstruction", "SGBM stereo depth"),
        ("🗿", "Mesh Generation",      "Poisson surface"),
        ("🎨", "Visualization",        "matplotlib + plotly"),
        ("🤖", "SDXL Inpainting",      "Baseline restoration"),
        ("🧬", "LoRA + ControlNet",    "Site-specific fine-tuning"),
    ]
    for icon, name, desc in stages:
        st.markdown(
            f'<div style="margin:0.35rem 0">'
            f'<span style="color:#c07840">{icon}</span> '
            f'<span style="color:#c09060;font-size:0.82rem">{name}</span><br>'
            f'<span style="color:#504030;font-size:0.74rem;padding-left:1.2rem">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("""
<div style="color:#403020;font-size:0.74rem;text-align:center">
Yashvardhan Gupta · Nikita Gupta<br>2026
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Hero Banner
# ═══════════════════════════════════════════════════════════════════════════════
raw_imgs   = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))
train_imgs = list(TRAIN_DIR.glob("*.png")) if TRAIN_DIR.exists() else []
lora_wts   = (OUT_DIR / "lora_weights" / "hampi_lora.pt").exists()

st.markdown("""
<div class="hero-wrap">
  <div class="hero-title">🏛️ HAMPI REVIVED</div>
  <div class="hero-sub">VIJAYANAGARA EMPIRE &nbsp;·&nbsp; 1336 – 1646 CE</div>
  <hr class="hero-divider">
  <div class="hero-tagline">
    "Stone meets Silicon — digitally resurrecting one of history's greatest architectural legacies<br>
    through computer vision, 3D reconstruction, and generative AI"
  </div>
  <div class="hero-badges">
    <span class="badge">UNESCO WORLD HERITAGE SITE</span>
    <span class="badge">KARNATAKA, INDIA</span>
    <span class="badge">1,600+ MONUMENTS</span>
    <span class="badge">COMPUTER VISION</span>
    <span class="badge">LORA FINE-TUNING</span>
    <span class="badge">CONTROLNET</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick stats
c1, c2, c3, c4, c5, c6 = st.columns(6)
metrics = [
    ("7", "Hampi Photographs"),
    (str(len(train_imgs)) or "—", "Training Crops"),
    ("34,544", "3D Cloud Points"),
    ("20", "Camera Poses"),
    ("8", "LoRA Rank"),
    ("✓ Ready" if lora_wts else "Training…", "LoRA Weights"),
]
for col, (val, lbl) in zip([c1,c2,c3,c4,c5,c6], metrics):
    with col:
        st.markdown(
            f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏰 History",
    "⚙️ Pipeline",
    "🖼️ Raw Images",
    "☁️ 3D Reconstruction",
    "🎨 SDXL Inpainting",
    "🧬 LoRA Restoration",
])


# ─── TAB 1: History ───────────────────────────────────────────────────────────
with tabs[0]:
    c_left, c_right = st.columns([3, 2], gap="large")

    with c_left:
        sec("THE VIJAYANAGARA EMPIRE")
        st.markdown("""
<p style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;color:#a08060">
Founded in 1336 CE on the banks of the Tungabhadra River in present-day Karnataka, the
Vijayanagara Empire stood for over three centuries as the last great Hindu kingdom of South India.
At its zenith under <b style="color:#d4a060">Krishnadevaraya (1509–1529)</b>, the capital city of Hampi
was one of the <i>largest cities in the world</i> — home to half a million souls, with markets
that stunned Portuguese and Persian visitors alike.
</p>
<p style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;color:#906050">
In 1565, a coalition of five Deccan Sultanates defeated the empire at the
<b style="color:#c07040">Battle of Talikota</b>. The victorious armies spent six months systematically
destroying the capital — shattering sculptures, toppling gopuram towers, and burning the great
markets. What remained was abandoned to the jungle for four centuries.
</p>
""", unsafe_allow_html=True)

        st.markdown("""
<div class="hist-quote">
  "The city of Vijayanagara is such that the pupil of the eye has never seen a place like it,
  and the ear of intelligence has never been informed that there existed anything to equal it
  in the world."
  <div class="attr">— Abdur Razzaq, Persian ambassador, 1443 CE</div>
</div>
""", unsafe_allow_html=True)

        sec("TIMELINE", "From founding to rediscovery")
        timeline_events = [
            ("1336 CE", "Foundation of the Empire",
             "Harihara I and Bukka Raya I establish Vijayanagara on the Tungabhadra. The city grows around the sacred Virupaksha Temple, which had stood since the 7th century."),
            ("1356 CE", "Sangama Dynasty Consolidates",
             "The empire expands to control most of the Deccan, becoming the primary power resisting the Bahmani Sultanate's southward incursions."),
            ("1424 CE", "Deva Raya II — Golden Age Begins",
             "Deva Raya II expands the empire to its greatest territorial extent. He is known for his religious tolerance and patronage of the arts and Sanskrit literature."),
            ("1509 CE", "Krishnadevaraya Ascends the Throne",
             "The greatest Vijayanagara ruler begins his reign. Under him the empire reaches its cultural apex — the Vittala Temple, Hazara Rama Temple, and numerous gopurams are built or expanded."),
            ("1520 CE", "Construction Peak",
             "Krishnadevaraya's building programme reaches its height. The iconic stone chariot at Vittala Temple and the musical pillars of the main mandapa are completed."),
            ("1565 CE", "Battle of Talikota — The Fall",
             "The Deccan Sultanates unite and defeat the imperial army. The capital is systematically destroyed over six months. Gopuram towers are toppled; market streets are demolished."),
            ("1800 CE", "Colonial Rediscovery",
             "British surveyor Colin Mackenzie documents the ruins for the first time. His maps reveal the extraordinary scale of what was lost."),
            ("1986 CE", "UNESCO World Heritage Site",
             "The Group of Monuments at Hampi is inscribed on the UNESCO World Heritage List, recognising its outstanding universal value."),
            ("2026 CE", "Hampi Revived",
             "This project uses computer vision and generative AI to digitally reconstruct missing architectural elements — the first computational pipeline purpose-built for Vijayanagara heritage."),
        ]

        st.markdown('<div class="timeline">', unsafe_allow_html=True)
        for year, title, body in timeline_events:
            st.markdown(f"""
<div class="tl-entry">
  <div class="tl-dot"></div>
  <div class="tl-year">{year}</div>
  <div class="tl-title">{title}</div>
  <div class="tl-body">{body}</div>
</div>
""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        sec("KEY MONUMENTS AT HAMPI")
        monuments = [
            ("Virupaksha Temple", "7th century CE onward",
             "The only continuously active temple at Hampi, dedicated to Shiva. Its 52-metre gopuram tower survives intact — one of the few complete examples of Vijayanagara tower architecture.",
             ["Active temple", "52m gopuram", "7th century"]),
            ("Vittala Temple Complex", "15th–16th century",
             "Contains the celebrated stone chariot (ratha) and the famous musical pillars — 56 hollow granite columns that produce musical notes when struck. Upper portions of the main gopuram are missing.",
             ["Stone chariot", "Musical pillars", "56 pillars"]),
            ("Hazara Rama Temple", "Early 15th century",
             "The royal chapel of the Vijayanagara kings. Its outer walls are covered in continuous bas-relief friezes narrating scenes from the Ramayana — over 1,000 carved panels.",
             ["Royal chapel", "Ramayana friezes", "1000+ carvings"]),
            ("Lotus Mahal", "16th century",
             "An elegant pavilion combining Islamic arched niches with Hindu corbelled domes. Its double-storeyed design and lotus-petal cornices are unique in Vijayanagara architecture.",
             ["Zenana enclosure", "Indo-Islamic", "Pavilion"]),
            ("Elephant Stables", "15th century",
             "Eleven domed chambers that once housed the royal war elephants. The alternating square and octagonal domes demonstrate the Vijayanagara synthesis of Hindu and Islamic architectural motifs.",
             ["11 chambers", "Royal elephants", "Indo-Islamic domes"]),
            ("Underground Shiva Temple", "16th century",
             "Partially submerged by the rising Tungabhadra river over centuries. The lingam remains flooded for much of the year — a haunting reminder of the site's abandonment.",
             ["Flooded annually", "Shiva lingam", "Subterranean"]),
        ]

        for name, period, desc, tags in monuments:
            tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
            st.markdown(f"""
<div class="monument-card">
  <div class="name">🏛️ {name}</div>
  <div style="color:#605040;font-size:0.74rem;margin-bottom:0.4rem;font-family:Cinzel,serif;letter-spacing:1px">{period}</div>
  <div class="desc">{desc}</div>
  <div style="margin-top:0.4rem">{tag_html}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    sec("ARCHITECTURAL VOCABULARY", "Key terms used in this restoration project")

    glossary = [
        ("Gopuram", "The ornate gateway tower of a South Indian temple complex. Multi-tiered structures rising above the entrance — the missing upper tiers (shikhara) are what this project aims to restore."),
        ("Shikhara", "The curvilinear or pyramidal superstructure above the main sanctuary or gateway. In Vijayanagara architecture, gopuram shikharas are typically barrel-vaulted at the top, with stucco figures on each tier."),
        ("Mandapa", "A pillared hall used for congregational worship, music, or royal audiences. Vijayanagara mandapas are distinguished by monolithic granite pillars with sculpted rearing horses and yalis (mythical lion-beasts)."),
        ("Kirtimukha", "Literally 'face of glory' — a demonic guardian face that appears at the apex of doorways, niches, and architectural frames throughout the complex. A key identifying motif of Vijayanagara carving vocabulary."),
        ("Ratha", "A chariot-shaped stone shrine or pavilion. The Vittala Temple's stone chariot (a ratha) is the most photographed object at Hampi — its wheels once rotated."),
        ("Yali", "A composite mythological creature — part lion, part elephant, part horse — that appears on pillars and friezes. Often depicted in rearing posture with a rider. A uniquely Vijayanagara sculptural invention."),
        ("Kalyana Mandapa", "A 'marriage hall' within the temple complex, used for the ritual celestial wedding of deities. Distinguished by elaborate ceiling carvings and larger floor space."),
        ("Pushkarini", "A sacred stepped tank (tank = reservoir) used for ritual bathing. Typically square in plan with symmetrical stairways descending on all four sides to the water level."),
    ]

    g_cols = st.columns(2)
    for i, (term, definition) in enumerate(glossary):
        with g_cols[i % 2]:
            st.markdown(f"""
<div class="arch-card">
  <div class="term">{term}</div>
  <div class="def">{definition}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    sec("WHY THIS PROJECT MATTERS")
    st.markdown("""
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-top:0.5rem">
  <div style="background:#110d08;border:1px solid #2a1808;border-radius:6px;padding:1.1rem">
    <div style="font-family:Cinzel,serif;color:#c07840;font-size:0.85rem;margin-bottom:0.5rem">📜 HERITAGE AT RISK</div>
    <div style="color:#706050;font-size:0.82rem;line-height:1.6">
    Of Hampi's 1,600 monuments, fewer than 50 retain complete superstructures.
    Climate change, vegetation encroachment, and tourist pressure accelerate decay.
    No complete visual record of most structures exists.
    </div>
  </div>
  <div style="background:#110d08;border:1px solid #2a1808;border-radius:6px;padding:1.1rem">
    <div style="font-family:Cinzel,serif;color:#c07840;font-size:0.85rem;margin-bottom:0.5rem">🔬 RESEARCH GAP</div>
    <div style="color:#706050;font-size:0.82rem;line-height:1.6">
    No computational pipeline exists for Deccan medieval architecture.
    Existing tools (RomArch, etc.) target European Gothic/Roman styles.
    Vijayanagara's distinctive vocabulary — yalis, kirtimukhas, barrel-vault shikharas —
    requires site-specific training data.
    </div>
  </div>
  <div style="background:#110d08;border:1px solid #2a1808;border-radius:6px;padding:1.1rem">
    <div style="font-family:Cinzel,serif;color:#c07840;font-size:0.85rem;margin-bottom:0.5rem">🤖 THIS PROJECT</div>
    <div style="color:#706050;font-size:0.82rem;line-height:1.6">
    First pipeline combining SfM 3D reconstruction with site-specific LoRA fine-tuning
    for Vijayanagara monuments. The goal: a tool that ASI conservators and heritage
    researchers can use to visualise and document missing structural elements.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─── TAB 2: Pipeline ──────────────────────────────────────────────────────────
with tabs[1]:
    sec("9-STAGE RECONSTRUCTION PIPELINE")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("""
<p style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;color:#a08060">
The pipeline is structured in two halves: classical computer vision reconstructs what
<i>physically exists</i> in the surviving stones; generative AI inpaints what
<i>should have existed</i> in the missing upper tiers — constrained by site-specific
fine-tuning to match Hampi's exact visual vocabulary.
</p>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<p style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;color:#806050">
The result is not hallucination — it is probabilistic reconstruction bounded by
geometry (SfM sparse cloud) and style (Hampi LoRA trained on real photographs),
with structural conditioning (ControlNet-Canny) preventing geometric inconsistency
at the seam between intact and generated regions.
</p>
""", unsafe_allow_html=True)

    st.markdown("---")
    pipeline_cards = [
        ("📥 Data Ingestion", "Wikimedia Commons scraper downloads Hampi monument photographs. A synthetic granite-ruin generator provides fallback images when the API rate-limits, ensuring the pipeline always has training data.", ["Wikimedia API", "Pillow", "CLAHE synthetic"]),
        ("🔬 Preprocessing", "CLAHE in LAB colourspace boosts carved-stone edge contrast without distorting hue. Non-Local Means removes sensor noise while preserving structural edges. Unsharp masking amplifies high-frequency relief detail (+177% Laplacian variance).", ["OpenCV", "CLAHE", "NLM denoising"]),
        ("🔍 Feature Extraction", "SIFT detects up to 5,000 scale- and rotation-invariant keypoints per image. FLANN with KD-tree finds approximate nearest neighbours; Lowe's ratio test (0.75) eliminates ambiguous matches.", ["SIFT", "FLANN", "Lowe ratio"]),
        ("📐 Structure from Motion", "Essential matrix estimated with RANSAC for each image pair. Camera pose recovered via SVD decomposition. Triangulation yields a sparse 3D point cloud from 20 registered cameras.", ["Essential matrix", "RANSAC", "Triangulation"]),
        ("☁️ Dense Reconstruction", "SGBM stereo disparity on consecutive image pairs generates per-pixel depth maps. Back-projection yields 33,261 combined dense points — a full spatial model of surviving structure.", ["SGBM stereo", "Back-projection", "Dense cloud"]),
        ("🗿 Mesh Generation", "Poisson surface reconstruction (Open3D) converts the point cloud into a watertight mesh. Exported as .ply for downstream rendering and analysis.", ["Open3D", "Poisson", "trimesh"]),
        ("📊 Visualization", "Matplotlib static quality dashboards and 3D scatter plots. Plotly interactive HTML point cloud — rotate, zoom, and filter in-browser.", ["matplotlib", "plotly", "interactive HTML"]),
        ("🤖 SDXL Inpainting", "Laplacian-variance boundary detection finds the damage horizon. SDXL fills the masked region. Cosine-feathered composite blends inpainted top with intact base.", ["SDXL", "gradio_client", "cosine blend"]),
        ("🧬 LoRA + ControlNet", "84 augmented training images fine-tune a rank-8 LoRA on the inpainting UNet. ControlNet-Canny conditions the generation on the structural edges of the surviving base. LAB colour matching corrects the seam.", ["DreamShaper-8", "ControlNet-Canny", "rank-8 LoRA", "LAB colour match"]),
    ]

    cols = st.columns(3)
    for i, (title, body, tags) in enumerate(pipeline_cards):
        with cols[i % 3]:
            tag_html = " ".join(pill(t) for t in tags)
            st.markdown(f"""
<div class="stage-card">
  <h4>{title}</h4>
  <p>{body}</p>
  <div style="margin-top:0.5rem">{tag_html}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    sec("KEY RESULTS SUMMARY")
    res_cols = st.columns(5)
    results = [
        ("1,283", "Sparse 3D Points"),
        ("33,261", "Dense 3D Points"),
        ("47", "Camera Pairs"),
        ("~12M", "LoRA Parameters"),
        ("~24 MB", "LoRA Weight File"),
    ]
    for col, (val, lbl) in zip(res_cols, results):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )


# ─── TAB 3: Raw Images ────────────────────────────────────────────────────────
with tabs[2]:
    sec("RAW HAMPI PHOTOGRAPHS",
        f"{len(raw_imgs)} photographs of Hampi monuments · Gopurams, mandapas, carved friezes")

    name_map = raw_name_map()
    if raw_imgs:
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(raw_imgs)):
            with cols[i % 4]:
                img = load_img(img_path)
                if img:
                    monument_name = name_map.get(img_path.stem, img_path.name)
                    st.image(img, use_container_width=True)
                    st.markdown(
                        f'<div class="img-label">{monument_name}</div>'
                        f'<div style="color:#3a2810;font-size:0.7rem;text-align:center">{img_path.name}</div>',
                        unsafe_allow_html=True
                    )
    else:
        st.warning("No raw images found in `data/raw/`. Run the data ingestion stage first.")

    st.markdown("---")
    sec("AUGMENTED TRAINING SET (LORA)",
        f"7 raw images × 12 augmentations each = **{len(train_imgs)} training crops**")

    st.markdown("""
<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:0.6rem;
background:#110d08;border:1px solid #1e1408;border-radius:6px;padding:1rem;margin-bottom:1rem">
""" + "".join([
        f'<div style="background:#2a1808;border-radius:4px;padding:0.3rem 0.5rem;font-size:0.72rem;color:#a06030">✓ {aug}</div>'
        for aug in ["Centre crop", "H-Flip", "CLAHE", "CLAHE+Flip",
                    "Bright 0.85×", "Bright 1.15×", "Contrast 1.2×",
                    "Sharpen 1.5×", "Random crop A", "Crop A+Flip",
                    "Random crop B", "Crop B+Flip"]
    ]) + "</div>", unsafe_allow_html=True)

    if train_imgs:
        sample = train_imgs[:12]
        cols = st.columns(6)
        for i, p in enumerate(sample):
            with cols[i % 6]:
                img = load_img(p)
                if img:
                    st.image(img, use_container_width=True)
                    st.markdown(f'<div class="img-label">{p.stem[-4:]}</div>', unsafe_allow_html=True)
        if len(train_imgs) > 12:
            st.caption(f"Showing 12 of {len(train_imgs)} crops. Run `python lora_pipeline.py` to regenerate.")
    else:
        st.info("Run `python lora_pipeline.py` to generate the augmented training set.")


# ─── TAB 4: 3D Reconstruction ─────────────────────────────────────────────────
with tabs[3]:
    sec("STRUCTURE FROM MOTION — SPARSE CLOUD")

    c1, c2 = st.columns(2)
    with c1:
        show_img(VIZ_DIR / "sparse_cloud_3d.png", "Sparse 3D point cloud (1,283 pts · 20 cameras)")
    with c2:
        show_img(VIZ_DIR / "topdown_view.png", "Top-down projection — camera positions visible")

    st.markdown("---")
    sec("FEATURE MATCHING")

    c1, c2 = st.columns(2)
    with c1:
        show_img(VIZ_DIR / "match_matrix.png", "FLANN match count matrix — 47 connected pairs")
    with c2:
        show_img(VIZ_DIR / "image_grid.png", "Sample preprocessed images entering SfM")

    st.markdown("---")
    sec("QUALITY DASHBOARD")
    show_img(VIZ_DIR / "quality_dashboard.png", "Reconstruction quality metrics across all images")

    st.markdown("---")
    sec("INTERACTIVE 3D POINT CLOUD")
    html_path = VIZ_DIR / "interactive_cloud.html"
    if html_path.exists():
        with open(html_path) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=False)
        st.caption("Interactive Plotly 3D cloud — rotate · zoom · hover for coordinates")
    else:
        st.info("Run `python pipeline.py` to generate the interactive cloud.")

    st.markdown("---")
    sec("RECONSTRUCTION METRICS")
    col1, col2, col3, col4 = st.columns(4)
    for col, (val, lbl) in zip([col1,col2,col3,col4], [
        ("20", "Registered Cameras"),
        ("1,283", "Sparse 3D Points"),
        ("33,261", "Dense 3D Points"),
        ("47", "Connected Image Pairs"),
    ]):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )


# ─── TAB 5: SDXL Inpainting ───────────────────────────────────────────────────
with tabs[4]:
    sec("GENERATIVE RESTORATION — SDXL BASELINE",
        "Damage boundary detection → masked inpainting → cosine-feathered composite")

    st.markdown("""
<p style="font-family:'EB Garamond',serif;font-size:1rem;line-height:1.8;color:#a08060">
The <b style="color:#c09060">target gopuram</b> (<code style="color:#806040">59b2b09ec5.jpg</code>)
is a Vijayanagara entrance gateway whose upper shikhara tiers have been lost.
A Laplacian-variance boundary detector identifies the damage horizon; everything above it
is masked and filled by SDXL inpainting via HuggingFace.
</p>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_img(OUT_DIR / "generative_reconstruction_inpaint.png",
                 "SDXL True Inpainting — 2×3 comparison panel")
    with c2:
        show_img(OUT_DIR / "generative_reconstruction_controlnet.png",
                 "ControlNet-guided inpainting — Canny edge conditioning")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        show_img(OUT_DIR / "generative_reconstruction_v2.png",
                 "SDXL v2 with feathered composite")
    with c2:
        show_img(OUT_DIR / "v1 result — FLUX text-to-image only, no structure conditioning.png",
                 "v1 — FLUX text-to-image only (no structure conditioning, for comparison)")

    st.markdown("---")
    sec("HOW THE SDXL INPAINTING WORKS")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Damage Boundary Detection**
- Laplacian variance computed in 30-px horizontal windows
- Boundary = row where variance drops fastest (sharpness loss = damage onset)
- Constrained to 15–50% of image height to avoid false positives at sky/ground

**Mask Construction**
- Binary mask: white (fill) above boundary, black (keep) below
- Cosine feathering ±40px around boundary smooths the seam
""")
    with c2:
        st.markdown("""
**Inpainting Model**
- SDXL via `diffusers/stable-diffusion-xl-inpainting`
- 30 inference steps · guidance scale 8.0 · strength 0.99
- Prompt anchors to Vijayanagara vocabulary + granite texture
- Unmasked pixels mathematically clamped — intact structure cannot change

**Composite**
- Inpainted top merged with original intact base
- Cosine-weighted alpha: 1.0 at top → 0.0 below boundary
- Change heatmap quantifies modification extent
""")

    st.markdown("---")
    sec("UPLOAD YOUR OWN IMAGE")
    uploaded = st.file_uploader(
        "Upload a gopuram or temple image to preview restoration target",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        c1, c2 = st.columns(2)
        with c1:
            st.image(img, caption="Uploaded image", use_container_width=True)
        with c2:
            st.markdown("""
**To run LoRA restoration on this image:**
```bash
# Save to data/raw/
cp your_image.jpg data/raw/

# Run restoration (uses cached LoRA weights)
IMG_PATH=data/raw/your_image.jpg python lora_pipeline.py
```
The result will appear in `outputs/lora_your_image.png`.
""")


# ─── TAB 6: LoRA Restoration ──────────────────────────────────────────────────
with tabs[5]:
    sec("LORA FINE-TUNING — VIJAYANAGARA ARCHITECTURAL VOCABULARY")

    st.markdown("""
<p style="font-family:'EB Garamond',serif;font-size:1.05rem;line-height:1.85;color:#a08060">
The SDXL baseline produces plausible but <i>generic</i> Dravidian completions — the stone colour,
tier proportions, and carving density don't match Hampi's specific Vijayanagara vocabulary.
A <b style="color:#d4a060">rank-8 LoRA</b> trained on real Hampi photographs, combined with
<b style="color:#d4a060">ControlNet-Canny</b> structural conditioning, anchors every generation
to the correct visual language while maintaining geometric coherence with the surviving base.
</p>
""", unsafe_allow_html=True)

    # LoRA status banner
    lora_pt = OUT_DIR / "lora_weights" / "hampi_lora.pt"
    if lora_pt.exists():
        sz_mb = lora_pt.stat().st_size / 1e6
        st.success(f"✓  LoRA weights trained and saved  ({sz_mb:.1f} MB)  ·  "
                   f"Rank {8}  ·  {500} training steps  ·  ControlNet-Canny enabled")
    else:
        st.warning(
            "LoRA weights not yet available. Training in progress or run:\n"
            "```bash\nsource venv/bin/activate && FORCE_RETRAIN=1 python lora_pipeline.py\n```\n"
            "Training takes ~2–4 hrs on Apple MPS (500 steps) / ~40 min on a GPU."
        )

    st.markdown("---")
    sec("THREE IMPROVEMENTS OVER THE BASELINE")

    imp_cols = st.columns(3)
    improvements = [
        ("🎯 Contour-Aware Mask",
         "Sky detection (HSV blue + overcast) per column finds where the building silhouette ends. "
         "The mask follows the actual structural edge rather than a flat horizontal cut — "
         "preventing the model from 'restoring' undamaged cornices.",
         ["HSV sky detection", "Per-column boundary", "Gaussian smooth"]),
        ("🏗️ ControlNet-Canny",
         "Canny edges from the preserved lower portion condition the inpainting UNet via "
         "ControlNet-Canny (scale 0.6). Generated tiers must be geometrically consistent with "
         "the surviving base — pillar alignment, arch widths, and cornice heights are anchored.",
         ["lllyasviel/sd15-canny", "Scale 0.6", "Preserved edges only"]),
        ("🎨 LAB Colour Matching",
         "After compositing, the generated region's LAB colour statistics are transferred "
         "to match the preserved base region. This corrects the colour temperature mismatch "
         "that made generated tops look 'pasted on'.",
         ["LAB colourspace", "Mean/std transfer", "Seam-corrected"]),
    ]
    for col, (title, body, tags) in zip(imp_cols, improvements):
        with col:
            tag_html = " ".join(pill(t) for t in tags)
            st.markdown(f"""
<div style="background:#130f0a;border:1px solid #2a1808;border-top:3px solid #8a6020;
border-radius:6px;padding:1.2rem;height:100%">
  <div style="font-family:Cinzel,serif;color:#d4a060;font-size:0.95rem;margin-bottom:0.6rem">{title}</div>
  <div style="color:#706050;font-size:0.82rem;line-height:1.6;margin-bottom:0.7rem">{body}</div>
  <div>{tag_html}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    sec("LORA ARCHITECTURE & TRAINING")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**What LoRA modifies**
- Injects trainable low-rank matrices into frozen attention projections: `to_k`, `to_q`, `to_v`, `to_out`
- Effective weight delta: `ΔW = (α/r) · B @ A`
- Base model weights **frozen** — only ~12M LoRA parameters train
- Rank r = 8 → ~1.4% of total attention parameters

**Why trained on the inpainting model directly**
- Training previously used the base UNet (4-channel conv_in)
- Now trained on the inpainting UNet (9-channel conv_in for masked conditioning)
- Ensures perfect state-dict compatibility at inference — no cross-architecture key mismatch
""")
    with c2:
        st.markdown("""
**Training configuration**

| Parameter | Value |
|---|---|
| Model | `Lykon/dreamshaper-8-inpainting` |
| LoRA rank r | **8** (was 4) |
| LoRA alpha α | **32.0** (was 16) |
| Training steps | **500** (was 80) |
| Batch size | 1 |
| Learning rate | 1 × 10⁻⁴ |
| LR schedule | Cosine annealing |
| Optimiser | AdamW |
| Dropout | 0.05 |
| Grad clip | 1.0 |
| Device | Apple MPS / CUDA |
""")

    st.markdown("---")
    sec("PER-IMAGE RESTORATION RESULTS",
        "Each raw image processed with the same LoRA weights · ControlNet-Canny · contour mask")

    name_map = raw_name_map()
    lora_outputs = sorted((OUT_DIR).glob("lora_*.png"))
    # Exclude the old generic one if it exists
    lora_outputs = [p for p in lora_outputs if p.stem != "lora_reconstruction"]

    if lora_outputs:
        for lora_path in lora_outputs:
            stem = lora_path.stem.replace("lora_", "")
            monument_name = name_map.get(stem, stem)
            raw_path = RAW_DIR / f"{stem}.jpg"
            if not raw_path.exists():
                raw_path = RAW_DIR / f"{stem}.png"

            st.markdown(
                f'<div style="font-family:Cinzel,serif;color:#c09050;font-size:0.88rem;'
                f'letter-spacing:2px;margin:1.2rem 0 0.4rem;border-left:3px solid #5a3010;'
                f'padding-left:0.8rem">{monument_name}</div>',
                unsafe_allow_html=True
            )

            r_col, l_col = st.columns(2)
            with r_col:
                st.markdown('<div class="compare-lbl">▲ ORIGINAL (DAMAGED)</div>', unsafe_allow_html=True)
                show_img(raw_path)
            with l_col:
                st.markdown('<div class="compare-lbl">▲ LORA RESTORATION (CONTROLNET + RANK-8)</div>', unsafe_allow_html=True)
                show_img(lora_path)

            st.markdown('<hr style="border-color:#1a1208;margin:0.5rem 0">', unsafe_allow_html=True)
    else:
        st.info(
            "LoRA restoration outputs will appear here once the pipeline completes.\n\n"
            "```bash\nsource venv/bin/activate && python lora_pipeline.py\n```"
        )

    st.markdown("---")
    sec("EVOLUTION OF APPROACHES")

    variants = [
        (OUT_DIR / "v1 result — FLUX text-to-image only, no structure conditioning.png",
         "v1: FLUX text-only", "No structural conditioning. Pure hallucination."),
        (OUT_DIR / "generative_reconstruction_inpaint.png",
         "v2: SDXL Inpainting", "Flat horizontal mask. No site-specific style."),
        (OUT_DIR / "generative_reconstruction_controlnet.png",
         "v3: SDXL + ControlNet", "Structural conditioning added. Generic style."),
        (OUT_DIR / "lora_59b2b09ec5.png",
         "v4: LoRA rank-8 + ControlNet", "Site-specific style + structural coherence + colour match."),
    ]
    vcols = st.columns(4)
    for col, (path, label, note) in zip(vcols, variants):
        with col:
            img = load_img(path)
            if img:
                st.image(img, use_container_width=True)
            else:
                st.markdown(f'<div style="background:#110d08;border:1px dashed #2a1808;border-radius:4px;height:160px;display:flex;align-items:center;justify-content:center;color:#3a2810;font-size:0.75rem">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="img-label">{label}</div>', unsafe_allow_html=True)
            st.caption(note)

    st.markdown("---")
    sec("WHAT COMES NEXT")
    next_cols = st.columns(3)
    roadmap = [
        ("🗃️ More Training Data",
         "Expand from 7 to 100+ raw images (ASI archives, systematic Wikimedia crawl, field photography) "
         "to give the LoRA enough signal to learn tier proportions and cornice profiles reliably."),
        ("🧩 Paired Damage Dataset",
         "Synthetic pairs: programmatically damage intact monument images to create (damaged, complete) "
         "supervision pairs. Teaches the reconstruction task directly, not just architectural style."),
        ("🌐 Multi-View NeRF Inpainting",
         "Use the SfM point cloud as a 3D constraint. Fill the missing volume in 3D space (SPIn-NeRF) "
         "rather than in a single 2D view — geometric consistency across all viewing angles."),
    ]
    for col, (title, body) in zip(next_cols, roadmap):
        with col:
            st.markdown(f"""
<div style="background:#130f0a;border:1px solid #2a1808;border-radius:6px;padding:1rem">
  <div style="font-family:Cinzel,serif;color:#c07840;font-size:0.85rem;margin-bottom:0.5rem">{title}</div>
  <div style="color:#706050;font-size:0.8rem;line-height:1.6">{body}</div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1.5rem 0 1rem">
  <div style="font-family:Cinzel,serif;color:#4a3020;font-size:0.78rem;letter-spacing:3px">
    HAMPI REVIVED · YASHVARDHAN GUPTA · NIKITA GUPTA · 2026
  </div>
  <div style="color:#2a1a10;font-size:0.72rem;margin-top:0.4rem">
    Computer Vision &nbsp;·&nbsp; Structure from Motion &nbsp;·&nbsp;
    Generative AI &nbsp;·&nbsp; LoRA Fine-Tuning &nbsp;·&nbsp; ControlNet
  </div>
  <div style="color:#1a1008;font-size:0.68rem;margin-top:0.3rem;font-style:italic">
    In memory of the city that once rivalled Rome
  </div>
</div>
""", unsafe_allow_html=True)
