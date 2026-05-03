"""
Hampi Revived — Streamlit Web Application

Showcases the full pipeline:
  • Raw imagery + preprocessing
  • 3D reconstruction (SfM, dense cloud, mesh)
  • Generative inpainting (SDXL baseline)
  • LoRA-conditioned restoration

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
    /* Dark stone palette */
    :root {
        --stone: #c8a97e;
        --rust:  #b85c38;
        --dark:  #1a1410;
        --mid:   #2d2520;
    }
    .main { background-color: #0f0d0b; }

    /* Hero banner */
    .hero {
        background: linear-gradient(135deg, #1a0f05 0%, #2d1810 40%, #1a1005 100%);
        border: 1px solid #4a3020;
        border-radius: 12px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 { font-size: 2.8rem; color: #e8c87a; margin: 0; letter-spacing: 3px; }
    .hero .sub { color: #a89070; font-size: 1.05rem; margin-top: 0.5rem; }
    .hero .tagline {
        color: #d4a060;
        font-size: 1.2rem;
        font-style: italic;
        margin-top: 1rem;
        border-top: 1px solid #3a2510;
        padding-top: 1rem;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin: 1rem 0; }
    .metric-card {
        flex: 1;
        background: #1e1810;
        border: 1px solid #3a2a18;
        border-radius: 8px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card .val { font-size: 2rem; font-weight: bold; color: #e8c060; }
    .metric-card .lbl { font-size: 0.8rem; color: #907050; margin-top: 0.3rem; }

    /* Stage cards */
    .stage-card {
        background: #1a1410;
        border-left: 4px solid #b85c38;
        border-radius: 0 8px 8px 0;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }
    .stage-card h4 { color: #e8b870; margin: 0 0 0.3rem 0; font-size: 0.95rem; }
    .stage-card p  { color: #907868; margin: 0; font-size: 0.82rem; line-height: 1.5; }

    /* Section headers */
    .section-header {
        color: #d4903a;
        font-size: 1.3rem;
        font-weight: bold;
        border-bottom: 2px solid #3a2010;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Pill badges */
    .pill {
        display: inline-block;
        background: #3a1a08;
        border: 1px solid #6a3818;
        border-radius: 20px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        color: #c07840;
        margin: 0.2rem;
    }

    /* Image caption override */
    .caption { color: #908070; font-size: 0.78rem; text-align: center; margin-top: 0.3rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #c09070; }
    .stTabs [aria-selected="true"] { color: #e8c060 !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #120e0a;
        border-right: 1px solid #2a1a10;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_img(path) -> Image.Image | None:
    p = Path(path)
    return Image.open(p).convert("RGB") if p.exists() else None


def show_img(path, caption="", use_column_width=True):
    img = load_img(path)
    if img:
        st.image(img, caption=caption, use_container_width=use_column_width)
    else:
        st.info(f"🔲 Not yet generated: `{Path(path).name}`")


def pill(text: str) -> str:
    return f'<span class="pill">{text}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏛️ Hampi Revived")
    st.markdown("---")
    st.markdown("""
**Digital resurrection of the**
**Vijayanagara Empire (1336–1646 CE)**

Using computer vision, 3D reconstruction,
and generative AI to restore ruins of one
of history's greatest architectural legacies.
""")
    st.markdown("---")
    st.markdown("### Pipeline Stages")
    stages = [
        ("📥", "Data Ingestion",       "Wikimedia + synthetic"),
        ("🔬", "Preprocessing",        "CLAHE · denoise · sharpen"),
        ("🔍", "Feature Extraction",   "SIFT + FLANN matching"),
        ("📐", "Structure from Motion","Essential matrix, RANSAC"),
        ("☁️", "Dense Reconstruction", "SGBM stereo depth"),
        ("🗿", "Mesh Generation",      "Poisson surface"),
        ("🎨", "Visualization",        "matplotlib + plotly"),
        ("🤖", "Generative Inpaint",   "SDXL inpainting"),
        ("🧬", "LoRA Pipeline",        "Fine-tuned restoration"),
    ]
    for icon, name, desc in stages:
        st.markdown(f"**{icon} {name}**")
        st.caption(desc)
    st.markdown("---")
    st.caption("Yashvardhan Gupta · Nikita Gupta · 2026")


# ═══════════════════════════════════════════════════════════════════════════════
# Hero Banner
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🏛️ HAMPI REVIVED</h1>
  <div class="sub">
    Vijayanagara Empire &nbsp;·&nbsp; 1336–1646 CE &nbsp;·&nbsp;
    Computer Vision &nbsp;+&nbsp; Generative AI
  </div>
  <div class="tagline">
    "Stone meets Silicon — digitally reconstructing one of history's greatest architectural legacies"
  </div>
</div>
""", unsafe_allow_html=True)

# Quick stats row
raw_imgs   = list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))
train_imgs = list(TRAIN_DIR.glob("*.png")) if TRAIN_DIR.exists() else []
lora_done  = (OUT_DIR / "lora_reconstruction.png").exists()
lora_wts   = (OUT_DIR / "lora_weights" / "hampi_lora.pt").exists()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Raw Images",       len(raw_imgs))
col2.metric("Training Crops",   len(train_imgs) or "—")
col3.metric("3D Points",        "34,544")
col4.metric("Camera Poses",     "20")
col5.metric("LoRA Weights",     "✓ Ready" if lora_wts else "Training…")


# ═══════════════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "🏠 Overview",
    "🖼️ Raw Images",
    "☁️ 3D Reconstruction",
    "🎨 Generative Inpainting",
    "🧬 LoRA Restoration",
])


# ─── TAB 1: Overview ──────────────────────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="section-header">Project Architecture</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("""
#### The Problem
The Hampi ruins (Karnataka, India) — a UNESCO World Heritage Site — preserve only the
lower tiers of many gopurams (tower gateways). The upper shikhara sections have eroded,
collapsed, or been vandalized over four centuries.

No digital model existed that combined:
- Photogrammetric **3D geometry** from field images
- **AI-driven completion** of the missing architectural elements
- **Site-specific fine-tuning** anchored to Vijayanagara vocabulary
""")

    with c2:
        st.markdown("""
#### The Approach
We built a **9-stage pipeline** that progressively adds information:

1. Classical CV reconstructs what **physically exists**
2. Generative AI inpaints what **should have existed**
3. LoRA fine-tuning ensures the generation matches **this specific site**

The result is not a hallucination — it is a probabilistic reconstruction
constrained by both geometry (SfM sparse cloud) and style (Hampi-trained LoRA).
""")

    st.markdown("---")
    st.markdown('<div class="section-header">Pipeline Stages</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    pipeline_cards = [
        ("📥 Data Ingestion",
         "Wikimedia Commons scraper downloads real Hampi photographs. A synthetic granite-ruin generator provides fallback images when the API rate-limits, ensuring the pipeline always has training data.",
         ["Wikimedia API", "Pillow", "CLAHE synthetic"]),
        ("🔬 Preprocessing",
         "CLAHE in LAB colorspace boosts carved-stone edge contrast without distorting colour. Non-Local Means removes sensor noise while preserving structural edges. Unsharp masking amplifies high-frequency relief detail (+177% Laplacian variance).",
         ["OpenCV", "CLAHE", "NLM denoising"]),
        ("🔍 Feature Extraction",
         "SIFT detects up to 5,000 scale- and rotation-invariant keypoints per image. FLANN with KD-tree finds approximate nearest neighbours; Lowe's ratio test (0.75) eliminates ambiguous matches.",
         ["SIFT", "FLANN", "Lowe ratio"]),
        ("📐 Structure from Motion",
         "Essential matrix estimated with RANSAC for each image pair. Camera pose recovered via decomposition into rotation + translation. Triangulation yields a sparse 3D point cloud of 1,283 points from 20 registered cameras.",
         ["Essential matrix", "RANSAC", "Triangulation"]),
        ("☁️ Dense Reconstruction",
         "SGBM (Semi-Global Block Matching) stereo disparity on consecutive image pairs generates per-pixel depth maps. Back-projection into 3D yields 33,261 combined dense points.",
         ["SGBM stereo", "Back-projection", "Dense cloud"]),
        ("🗿 Mesh Generation",
         "Poisson surface reconstruction (Open3D, when available) or trimesh convex-hull fallback converts the point cloud into a watertight mesh. Exported as .ply for downstream use.",
         ["Open3D", "Poisson", "trimesh"]),
        ("📊 Visualization",
         "Matplotlib produces static quality dashboards and 3D scatter plots. Plotly generates an interactive HTML point cloud that can be rotated, zoomed, and filtered in-browser.",
         ["matplotlib", "plotly", "interactive HTML"]),
        ("🤖 SDXL Inpainting",
         "A Laplacian-variance boundary detector identifies the damage horizon on each gopuram. The mask above the boundary is filled by SDXL via HuggingFace Space. A cosine-feathered composite blends inpainted top + intact base.",
         ["SDXL", "gradio_client", "cosine blend"]),
        ("🧬 LoRA Restoration",
         "96 augmented training images (CLAHE, flips, brightness, random crops) fine-tune a LoRA (rank 8) on DreamShaper-8's UNet attention layers. The trained LoRA is transferred to the DreamShaper-8-inpainting model, anchoring all completions to Vijayanagara architectural vocabulary.",
         ["DreamShaper-8", "PEFT LoRA", "MPS training"]),
    ]

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
    st.markdown('<div class="section-header">Key Results</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Sparse cloud points",  "1,283")
    mc2.metric("Dense cloud points",   "33,261")
    mc3.metric("LoRA parameters",      "~4M / 859M")
    mc4.metric("LoRA model size",      "~16 MB")


# ─── TAB 2: Raw Images ────────────────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">Raw Hampi Photographs</div>', unsafe_allow_html=True)
    st.caption(
        f"{len(raw_imgs)} photographs of Hampi monuments · "
        "Gopurams, pillared halls, carved friezes, and fortification walls"
    )

    if raw_imgs:
        cols = st.columns(4)
        for i, img_path in enumerate(sorted(raw_imgs)):
            with cols[i % 4]:
                img = load_img(img_path)
                if img:
                    st.image(img, caption=img_path.name, use_container_width=True)
    else:
        st.warning("No raw images found in `data/raw/`. Run the data ingestion stage first.")

    st.markdown("---")
    st.markdown('<div class="section-header">Augmented Training Set (LoRA)</div>', unsafe_allow_html=True)
    st.caption(
        f"8 raw images × 12 augmentations each = **{len(train_imgs)} training crops** · "
        "Horizontal flips · CLAHE · Brightness/contrast · Random crops"
    )

    if train_imgs:
        # Show a 4-column sample (first 12)
        sample = train_imgs[:12]
        cols = st.columns(4)
        for i, p in enumerate(sample):
            with cols[i % 4]:
                img = load_img(p)
                if img:
                    st.image(img, caption=p.stem, use_container_width=True)
        if len(train_imgs) > 12:
            st.caption(f"… and {len(train_imgs) - 12} more. Run `python lora_pipeline.py` to regenerate.")
    else:
        st.info("Run `python lora_pipeline.py` to generate the augmented training set.")


# ─── TAB 3: 3D Reconstruction ─────────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">Structure from Motion — Sparse Cloud</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_img(VIZ_DIR / "sparse_cloud_3d.png", "Sparse 3D point cloud (1,283 points · 20 cameras)")
    with c2:
        show_img(VIZ_DIR / "topdown_view.png", "Top-down projection — camera positions visible")

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Matching</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        show_img(VIZ_DIR / "match_matrix.png", "FLANN match count matrix — 47 connected pairs")
    with c2:
        show_img(VIZ_DIR / "image_grid.png", "Sample preprocessed images entering SfM")

    st.markdown("---")
    st.markdown('<div class="section-header">Quality Dashboard</div>', unsafe_allow_html=True)
    show_img(VIZ_DIR / "quality_dashboard.png", "Reconstruction quality metrics across all images")

    st.markdown("---")
    st.markdown('<div class="section-header">Interactive 3D Point Cloud</div>', unsafe_allow_html=True)
    html_path = VIZ_DIR / "interactive_cloud.html"
    if html_path.exists():
        with open(html_path) as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600, scrolling=False)
        st.caption("Interactive Plotly 3D cloud — rotate · zoom · hover for coordinates")
    else:
        st.info("Interactive cloud not yet generated. Run `python pipeline.py` first.")

    st.markdown("---")
    st.markdown('<div class="section-header">Reconstruction Metrics</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Registered cameras", "20")
    col2.metric("Sparse points",      "1,283")
    col3.metric("Dense points",       "33,261")
    col4.metric("Image pairs",        "47")


# ─── TAB 4: Generative Inpainting ─────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">Generative Restoration — Baseline SDXL</div>', unsafe_allow_html=True)
    st.markdown("""
The **target gopuram** (`59b2b09ec5.jpg`) is a Vijayanagara entrance gateway whose upper
shikhara tiers have been lost. A Laplacian-variance boundary detector finds the damage horizon;
everything above it is masked and filled by SDXL inpainting.
""")

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
                 "v1 — FLUX text-to-image only (no structural conditioning, baseline)")

    st.markdown("---")
    st.markdown('<div class="section-header">Inpainting Pipeline Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Damage Boundary Detection**
- Laplacian variance computed in 30-pixel horizontal windows
- Boundary = row where variance drops fastest (sharpness loss = damage onset)
- Constrained to 15–50% of image height

**Mask Construction**
- Binary mask: white (fill) above boundary, black (keep) below
- Cosine feathering ±40px around boundary smooths the seam
""")
    with c2:
        st.markdown("""
**Inpainting**
- SDXL via `diffusers/stable-diffusion-xl-inpainting` HF Space
- 30 inference steps · guidance scale 8.0 · strength 0.99
- Prompt anchors to Vijayanagara vocabulary + granite texture

**Composite**
- Inpainted top merged with original intact base
- Cosine-weighted alpha: 1.0 at top → 0.0 below boundary
- Change heatmap quantifies modification extent
""")

    st.markdown("---")
    st.markdown('<div class="section-header">Run Inpainting on a Custom Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a gopuram / temple image to inpaint",
        type=["jpg", "jpeg", "png"],
    )
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", width=400)
        st.info(
            "To run inpainting: save the image to `data/raw/`, then run\n"
            "```bash\nIMG_PATH=data/raw/<your_file.jpg> python generative_reconstruct.py\n```"
        )


# ─── TAB 5: LoRA Restoration ──────────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">LoRA Fine-Tuning — Hampi Architectural Vocabulary</div>',
                unsafe_allow_html=True)

    st.markdown("""
The SDXL baseline produces plausible but **generic** Dravidian completions — the stone
colour, tier proportions, and carving density don't match Hampi's specific
Vijayanagara vocabulary. A **LoRA** (Low-Rank Adaptation) trained on the actual
Hampi image dataset anchors all generations to the right visual language.
""")

    # LoRA status
    lora_pt   = OUT_DIR / "lora_weights" / "hampi_lora.pt"
    lora_img  = OUT_DIR / "lora_reconstruction.png"

    if lora_pt.exists():
        sz_mb = lora_pt.stat().st_size / 1e6
        st.success(f"✓ LoRA weights trained and saved ({sz_mb:.1f} MB)")
    else:
        st.warning(
            "LoRA weights not yet available. Run the training pipeline:\n"
            "```bash\nsource venv/bin/activate\npython lora_pipeline.py\n```\n"
            "Training takes ~30–60 min on Apple MPS / ~10 min on a GPU."
        )

    st.markdown("---")
    st.markdown('<div class="section-header">LoRA Architecture</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**What LoRA modifies**
- Injects trainable low-rank matrices A (r×d) and B (d×r) into each
  attention projection: `to_k`, `to_q`, `to_v`, `to_out`
- Effective weight delta: `ΔW = α/r · B @ A`
- Base model weights **frozen** — only LoRA matrices train
- Rank r = 8 → 4M trainable / 859M total parameters (0.5%)
""")
    with c2:
        st.markdown("""
**Training setup**
- Base model: `Lykon/dreamshaper-8` (SD-1.5 fine-tune)
- Inpaint model: `Lykon/dreamshaper-8-inpainting` (same UNet arch)
- Training: 600 steps · cosine LR schedule · batch 1
- Augmentation: 8 images → 96 crops (CLAHE, flips, brightness, crops)
- Device: Apple MPS (M-series) · ~30–60 min
- LoRA α = 32 · dropout 0.05 · AdamW lr = 1e-4
""")

    st.markdown("---")
    st.markdown('<div class="section-header">LoRA Transfer to Inpainting</div>', unsafe_allow_html=True)
    st.markdown("""
The LoRA is trained on the **base** DreamShaper-8 model, then the attention-layer
weights are transferred to the **inpainting** DreamShaper-8 model. Both share identical
cross-attention projections — only the `conv_in` layer differs (4 vs 9 channels for
masked-image conditioning), and LoRA does not touch `conv_in`. This makes the transfer
exact with zero architecture mismatch.
""")

    st.markdown("---")
    st.markdown('<div class="section-header">LoRA Reconstruction Result</div>',
                unsafe_allow_html=True)

    if lora_img.exists():
        show_img(lora_img,
                 "End-to-end LoRA result — DreamShaper-8 + Hampi LoRA · 600 steps · rank 8")
    else:
        st.info(
            "The LoRA reconstruction figure will appear here once training completes.\n\n"
            "If the pipeline is running, check progress:\n"
            "```bash\ntail -f /tmp/lora_pipeline_run.log\n```"
        )

    st.markdown("---")
    st.markdown('<div class="section-header">Why LoRA over Prompt Engineering</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**Prompt-only baseline**
- Generic "Dravidian temple" output
- No site-specific stone colour
- Tier proportions don't match Hampi
- Carving density is hallucinated
""")
    with c2:
        st.markdown("""
**LoRA advantage**
- Trained on *actual Hampi* photography
- Sandstone oxidation colour learned
- Vijayanagara cornice profiles learned
- Carving density from real data
""")
    with c3:
        st.markdown("""
**Comparison to ControlNet**
- ControlNet conditions on *structure*
- LoRA conditions on *style + texture*
- Ideal: ControlNet + LoRA combined
- Next step: IP-Adapter for reference
""")

    # Show existing comparison outputs for reference
    st.markdown("---")
    st.markdown('<div class="section-header">All Reconstruction Variants — Side by Side</div>',
                unsafe_allow_html=True)
    variants = [
        (OUT_DIR / "v1 result — FLUX text-to-image only, no structure conditioning.png",
         "v1: FLUX text-only (no structure)"),
        (OUT_DIR / "generative_reconstruction_inpaint.png",
         "v2: SDXL True Inpainting"),
        (OUT_DIR / "generative_reconstruction_controlnet.png",
         "v3: SDXL + ControlNet"),
        (OUT_DIR / "lora_reconstruction.png",
         "v4: DreamShaper + LoRA (current)"),
    ]
    vcols = st.columns(4)
    for col, (path, label) in zip(vcols, variants):
        with col:
            img = load_img(path)
            if img:
                st.image(img, caption=label, use_container_width=True)
            else:
                st.markdown(f"*{label}*")
                st.caption("Not yet generated")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#504030; font-size:0.8rem; padding:1rem 0">
  Hampi Revived · Yashvardhan Gupta · Nikita Gupta · 2026<br>
  <span style="color:#3a2a18">
    Computer Vision · Structure from Motion · Generative AI · LoRA Fine-Tuning
  </span>
</div>
""", unsafe_allow_html=True)
