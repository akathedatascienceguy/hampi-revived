# Hampi Revived — Technical Reference

**Project:** Hampi Revived — Digital Reconstruction of the Vijayanagara Empire  
**Authors:** Yashvardhan Gupta · Nikita Gupta  
**Date:** 2026  

---

## Overview

Hampi (Karnataka, India) is one of the largest and most important medieval cities ever built — capital of the Vijayanagara Empire from 1336 to 1646 CE. Today it is a UNESCO World Heritage Site, but centuries of abandonment, Deccan Sultanate sacking, and monsoon erosion have reduced many of its 1,600 surviving monuments to partial ruins. Gopurams (tower gateways), mandapas (columned halls), and ratha shrines stand with their upper tiers missing or collapsed.

This project builds a full **9-stage pipeline** that uses classical computer vision to reconstruct what physically exists, and generative AI to complete what should have existed — anchored to Vijayanagara's specific architectural vocabulary through LoRA fine-tuning.

---

## Repository Structure

```
hampi-revived/
├── pipeline.py                   # Main CLI orchestrator (stages 1–8)
├── generative_reconstruct.py     # SDXL baseline inpainting
├── lora_pipeline.py              # End-to-end LoRA fine-tune + inpainting
├── app.py                        # Streamlit web application
├── src/
│   ├── data_ingestion.py         # Wikimedia download + synthetic fallback
│   ├── preprocessing.py          # CLAHE + denoise + sharpen
│   ├── feature_extraction.py     # SIFT detection + FLANN matching
│   ├── sfm.py                    # Essential matrix, RANSAC, triangulation
│   ├── dense_reconstruction.py   # SGBM stereo depth → dense cloud
│   ├── mesh.py                   # Poisson surface / trimesh fallback
│   ├── visualization.py          # matplotlib static + plotly interactive
│   └── groq_analysis.py          # Groq vision + text AI analysis
├── data/
│   ├── raw/                      # 8 Hampi monument photographs
│   └── lora_train/               # 96 augmented training crops (auto-generated)
├── outputs/
│   ├── visualizations/           # SfM plots, point cloud renders, dashboards
│   ├── meshes/                   # .ply point cloud / mesh files
│   ├── lora_weights/             # hampi_lora.pt (trained LoRA deltas)
│   ├── lora_reconstruction.png   # Final LoRA restoration figure
│   └── generative_reconstruction_*.png  # Baseline inpainting results
└── docs/
    ├── technical.md              # This document
    └── email_parnavi_bangar.md   # Collaboration outreach draft
```

---

## Stage 1 — Data Ingestion (`src/data_ingestion.py`)

Wikimedia Commons is queried for images of Hampi monuments using keyword searches over the MediaWiki API. Images are downloaded, deduplicated by file hash, and stored in `data/raw/`.

When Wikimedia rate-limits or returns fewer than the minimum required images, a **synthetic generator** creates plausible granite-ruin scenes: procedural granite texture (Perlin noise + sandstone colour palette), randomised pillar silhouettes, and simulated shadow occlusion at architectural boundaries. These synthetic images are flagged and excluded from the LoRA training set.

---

## Stage 2 — Preprocessing (`src/preprocessing.py`)

Three sequential operations prepare each image for feature extraction and generative use.

### 2.1 CLAHE — Contrast Limited Adaptive Histogram Equalization

Operates in **LAB colour space**. The L (luminance) channel is split into an 8×8 tile grid. Within each tile, a local histogram is computed and clipped at `clipLimit=2.0` — pixels above the threshold are redistributed uniformly, preventing noise amplification while boosting local contrast at carved-stone edges.

```
L_enhanced = CLAHE(L, clip=2.0, tileGrid=8×8)
```

**Why LAB:** Applying histogram equalization directly to RGB channels distorts colour. LAB separates luminance from chrominance (A, B channels), so sandstone hues are preserved while architectural edge contrast increases.

### 2.2 Non-Local Means Denoising

`fastNlMeansDenoisingColored` with `h=6.0`, template window 7×7, search window 21×21.

For each pixel **p**, a 7×7 patch is compared against all patches within a 21×21 neighbourhood. The denoised value is a patch-similarity-weighted average:

```
NLM(p) = Σ_q  w(p,q) · I(q)
w(p,q) = exp(−‖P(p) − P(q)‖² / h²)
```

High-similarity patches receive exponentially higher weight. This preserves structural edges (edge patches are dissimilar to flat-region patches, so they get low weight and are not averaged away) while removing sensor noise.

### 2.3 Unsharp Masking

```
output = 1.6 × original − 0.6 × GaussianBlur(σ=3)
```

The Gaussian blur removes high-frequency content. Subtracting it isolates edges and texture. Adding these high-frequency components back amplified sharpens carved-stone relief.

**Measured effect:** Laplacian variance (sharpness proxy) increased by **+177%** on the gopuram image (1,795 → 4,978).

---

## Stage 3 — Feature Extraction (`src/feature_extraction.py`)

### 3.1 SIFT — Scale-Invariant Feature Transform

For each image, SIFT detects up to 5,000 keypoints through:

1. **Gaussian scale-space pyramid** at multiple σ levels per octave
2. **Difference-of-Gaussian (DoG)** between adjacent scales
3. **Local extrema** in DoG → candidate keypoints, scale-stable by construction
4. **Low-contrast rejection** (threshold 0.04) and **edge filtering** (Harris corner ratio)
5. **Orientation assignment** from 36-bin gradient histogram in 16×16 neighbourhood
6. **128-dim descriptor**: 4×4 grid of 8-bin gradient histograms, L2-normalised

SIFT is rotation-, scale-, and partial-illumination-invariant — essential for Hampi's carved-granite surfaces photographed under varying sunlight, angle, and distance.

| Metric | Value |
|---|---|
| Mean keypoints per image | 4,123 |
| Min keypoints (blurry/dark) | 406 |
| Max keypoints | 5,001 |

### 3.2 FLANN Matching + Lowe's Ratio Test

FLANN (Fast Library for Approximate Nearest Neighbours) uses a KD-tree index (5 trees, 50 checks) to find the 2 nearest descriptor matches for each keypoint. Lowe's ratio test retains a match only if:

```
distance(best_match) / distance(second_best_match) < 0.75
```

This rejects ambiguous correspondences — a match is accepted only when one candidate is clearly better than all others.

| Metric | Value |
|---|---|
| Connected image pairs (≥20 matches) | 47 |
| Mean good matches per pair | 32.7 |
| Max matches per pair | 80 |

---

## Stage 4 — Structure from Motion (`src/sfm.py`)

### 4.1 Essential Matrix Estimation

Given matched point pairs (p₁, p₂) between two images with calibration matrix **K** (estimated from EXIF focal length or default f = max(width, height)):

```
x₂ᵀ E x₁ = 0,    E = [t]× R
```

**E** is the 3×3 Essential Matrix encoding both the rotation **R** and translation **t** between cameras. Estimated using the 5-point algorithm inside **RANSAC** (1,000 iterations, reprojection threshold 1.0 px) to robustly reject outlier matches from repetitive carved patterns.

### 4.2 Camera Pose Recovery

**R** and **t** are recovered from **E** via SVD decomposition. SVD yields 4 candidate (R, t) solutions; the correct one is selected by the cheirality condition (all triangulated points in front of both cameras):

```
E = U Σ Vᵀ
R ∈ {UWVᵀ, UWᵀVᵀ},   t ∈ {U[:,2], −U[:,2]}
```

### 4.3 Triangulation

3D point **X** is recovered from two camera projections via the Direct Linear Transform:

```
λ₁ x₁ = P₁ X,   λ₂ x₂ = P₂ X
→ [x₁ × P₁; x₂ × P₂] X = 0,   solved via SVD
```

**Sparse reconstruction results:**

| Metric | Value |
|---|---|
| 3D points triangulated | 1,283 (combined) |
| Camera poses registered | 20 |
| Valid image pairs | 47 |

---

## Stage 5 — Dense Reconstruction (`src/dense_reconstruction.py`)

SGBM (Semi-Global Block Matching) computes a dense disparity map between consecutive image pairs by minimising an energy function over all pixels simultaneously:

```
E(D) = Σ_p C(p, D_p) + Σ_{q∈N_p} [P₁·|D_p−D_q|=1 + P₂·|D_p−D_q|>1]
```

- **C(p, D_p)**: data cost — pixel dissimilarity at disparity D
- **P₁ = 8**: small disparity changes (smooth surfaces) penalised lightly
- **P₂ = 32**: large jumps (depth discontinuities at carved edges) penalised heavily

Disparity is back-projected to 3D: `Z = f·B/d` where f = focal length, B = stereo baseline, d = disparity.

**Dense reconstruction results:**

| Metric | Value |
|---|---|
| Dense points generated | 33,261 |
| Total cloud (sparse + dense) | 34,544 |

---

## Stage 6 — Mesh Reconstruction (`src/mesh.py`)

**Open3D Poisson surface reconstruction** fits a watertight triangle mesh by solving the Poisson equation over the oriented point cloud:

```
∇²χ = ∇·V
```

where **V** is the vector field defined by point normals, **χ** is an indicator function, and the isosurface is extracted at the learned level. Octree depth = 9. If Open3D is unavailable (Python > 3.12), a `trimesh` convex-hull fallback is used.

**Output:** `outputs/meshes/hampi_cloud_trimesh.ply`

---

## Stage 7 — Visualization (`src/visualization.py`)

- **Matplotlib:** static quality dashboard, 3D scatter plots of sparse/dense cloud, top-down projection showing camera positions
- **Plotly:** interactive HTML point cloud (`outputs/visualizations/interactive_cloud.html`) — full 3D rotation, zoom, and hover in-browser

---

## Stage 8 — Generative Inpainting — SDXL Baseline (`generative_reconstruct.py`)

### 8.1 Latent Diffusion Foundation

SDXL encodes images into a compressed latent space via a VAE:

```
z = E(x),   z ∈ ℝ^(H/8 × W/8 × 4)
```

An 8× spatial compression reduces a 768×512 image to a 96×64×4 latent. A U-Net denoiser ε_θ(z_t, t, c) is trained to predict the added noise at timestep t, conditioned on text embedding **c**. At inference, starting from Gaussian noise z_T, the model iteratively denoises guided by classifier-free guidance at scale 8.0:

```
ε_guided = ε_uncond + 8.0 × (ε_cond − ε_uncond)
```

### 8.2 Inpainting Mechanism

The inpainting variant conditions the U-Net on both the masked image and the binary mask channel concatenated to the noisy latent (9 total input channels vs. 4 for the base model). At each denoising step, **unmasked pixels are replaced with their noised originals** — the intact lower structure is mathematically clamped and cannot be modified. Only the masked region (damaged top) is free to evolve.

This is why the doorway, flanking carved walls, and stone colour in the output are pixel-identical to the input.

### 8.3 Damage Boundary Detection

A Laplacian variance sliding window (30-row bands) scans the image top-to-bottom:

```
var_r = Var(|∇²I|[r:r+30, :])
```

The row with the steepest drop in variance (largest negative first-difference in the upper half) marks the transition from "rich carved detail" (intact structure) to "rubble / missing" (damaged zone). This boundary is constrained to 15–50% of image height to avoid false positives at the sky/wall or ground transitions.

**Detected boundary on the target gopuram:** row 330/800 (41% from top).

### 8.4 Cosine-Feathered Compositing

To blend the inpainted top with the original intact base, a cosine-weighted alpha avoids the linear seam artefact:

```python
alpha(r) = 0.5 × (1 + cos(π × t))    # t ∈ [0,1] across ±40px blend band
output(r) = inpainted(r) × alpha(r) + original(r) × (1 − alpha(r))
```

The S-curve schedule is perceptually smooth because the eye is more sensitive to linear discontinuities than sinusoidal gradients.

---

## Stage 9 — LoRA Fine-Tuning (`lora_pipeline.py`)

The SDXL baseline produces plausible but **generic** Dravidian completions. The stone colour, tier proportions, and carving density do not match Hampi's specific Vijayanagara vocabulary. A **LoRA (Low-Rank Adaptation)** trained on the actual Hampi photographs anchors every generation to the correct visual language.

### 9.1 Data Augmentation

Eight raw Hampi photographs are augmented to 96 training crops:

| Augmentation | Count per image |
|---|---|
| Centre square crop (512×512) | 1 |
| Horizontal flip | 1 |
| CLAHE enhanced | 1 |
| CLAHE + flip | 1 |
| Brightness 0.85× | 1 |
| Brightness 1.15× | 1 |
| Contrast 1.2× | 1 |
| Sharpness 1.5× | 1 |
| Random crop A | 1 |
| Random crop A + flip | 1 |
| Random crop B | 1 |
| Random crop B + flip | 1 |

**Total: 12 augmentations × 8 images = 96 training samples**

All crops receive the same descriptive caption:
```
"ancient Hampi Vijayanagara carved stone temple ruins, granite gopuram,
 ornate carved pillars and friezes, historical India, photorealistic"
```

### 9.2 LoRA Architecture

LoRA injects trainable low-rank matrices into the frozen attention projections of the U-Net. For each targeted weight matrix **W₀**:

```
W = W₀ + ΔW = W₀ + (α/r) · B @ A
```

where:
- **A ∈ ℝ^(r×d)** — down-projection (randomly initialised)
- **B ∈ ℝ^(d×r)** — up-projection (zero-initialised, so ΔW = 0 at start)
- **r** = rank (8 in this pipeline — doubled from initial rank-4 to increase style capacity)
- **α** = scaling factor (32.0, set to 4×r per standard practice)

Targeted modules: `to_k`, `to_q`, `to_v`, `to_out.0` — the four attention projections in every transformer block of the U-Net.

**Parameter count:**
- Trainable LoRA parameters: **~12M** at rank 8 (~1.4% of attention parameters)
- Frozen base parameters: **859,520,964**
- LoRA weight file size: **~24 MB** vs. ~2 GB for the full model

### 9.3 Training Setup

| Hyperparameter | Value | Notes |
|---|---|---|
| Training model | `Lykon/dreamshaper-8-inpainting` | Trained directly on inpainting UNet (not base model) |
| Inpainting model | `Lykon/dreamshaper-8-inpainting` | Same model — ensures state-dict compatibility |
| LoRA rank r | **8** | Was 4; higher rank captures more style nuance |
| LoRA alpha α | **32.0** | Was 16; = 4×r per standard scaling practice |
| Training steps | **500** | Was 80; ~6× more exposure to Hampi imagery |
| Batch size | 1 | |
| Learning rate | 1×10⁻⁴ | |
| LR schedule | Cosine annealing | |
| Optimiser | AdamW | |
| LoRA dropout | 0.05 | |
| Gradient clipping | 1.0 | |
| Image size | 512×512 | |
| Device | Apple MPS (M-series) | |

**Why train on the inpainting model directly:** The previous design trained the LoRA on the 4-channel base UNet (`dreamshaper-8`) and transferred weights to the 9-channel inpainting UNet. This caused a state-dict mismatch — the inpainting UNet has additional LoRA-injectable layers that were not present during training, producing `-430/256` loaded tensors (a negative number indicating more expected keys than saved). By training on the inpainting model from the start, the saved LoRA state dict is architecturally identical to the inference model.

**Memory architecture:** VAE and text encoder run on CPU (frozen, no-grad), freeing MPS memory exclusively for the UNet forward-backward pass. During training, a fully-masked latent (mask = ones, masked_latent = zeros) is concatenated to the noisy latent to form the 9-channel inpainting input — teaching the model the reconstruction task without requiring paired damage/complete supervision.

**Gradient checkpointing:** enabled on the UNet — recomputes activations during backprop rather than caching them, trading 30% compute overhead for 40% peak memory reduction.

### 9.4 Contour-Aware Damage Mask

The original pipeline used a flat horizontal mask determined by Laplacian variance row-scanning. This over-inpainted intact regions (e.g. sky-only patches above an already-complete cornice) and under-inpainted asymmetrically damaged zones. The replacement uses per-column sky boundary detection:

```python
def _make_damage_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Detect blue sky and overcast/white sky separately
    sky = cv2.bitwise_or(
        cv2.inRange(hsv, [85, 20, 80],  [135, 255, 255]),  # blue
        cv2.inRange(hsv, [0,  0,  180], [180,  35, 255]),  # overcast
    )
    sky = cv2.dilate(sky, np.ones((7,7)), iterations=2)

    # Per-column: count contiguous sky rows from the top
    col_top = [first_non_sky_row(sky[:, col]) for col in range(w)]

    # Smooth (moving average width=61) and clamp to 5–65% of image height
    col_top = np.convolve(col_top, np.ones(61)/61, mode='same').clip(h*0.05, h*0.65)

    # Build per-pixel mask and Gaussian-blur the edge
    mask = np.zeros((h, w), uint8)
    for col in range(w):
        mask[:col_top[col], col] = 255
    return int(np.median(col_top)), GaussianBlur(mask, 51)
```

**Effect:** For a mostly-intact temple, the sky boundary may be very high (small mask); for a truncated gopuram where tiers are missing, the sky intrudes further down through gaps in the structure (larger, correctly shaped mask).

### 9.5 ControlNet-Canny Structural Conditioning

Without geometric guidance, the inpainting model hallucinates tiers that don't match the surviving structure's pillar alignment, arch width, or cornice height. ControlNet-Canny provides this guidance:

```python
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    INPAINT_MODEL, controlnet=controlnet
)

# Canny on original image; zero out edges inside the mask
# so the model is unconstrained in the generation region
# but anchored by preserved structure edges below
canny = cv2.Canny(gray_orig, 80, 200)
canny[mask > 127] = 0
control_image = Image.fromarray(np.stack([canny]*3, axis=-1))

result = pipe(
    ...,
    control_image=control_image,
    controlnet_conditioning_scale=0.6,
)
```

**Conditioning scale 0.6:** Strong enough to anchor geometric continuity (pillar spacing, wall thickness) but not so strong that it forces the model to exactly reproduce the damaged structure's partial edges in the generated region.

**Zero-masking inside the inpaint region:** The Canny edges inside the mask come from the original damaged image (mostly sky or partial rubble). Zeroing them out removes spurious structural guidance in the generation zone, while preserving the full edge map below the mask boundary — the most valuable source of geometric constraint.

### 9.6 Pipeline Reuse Across Images

The inpainting pipeline (1.5–2 GB loaded model) is now built **once** and passed to each per-image inference call, rather than reloading per image. This eliminates 6 redundant model loads and ~30% of total wall-clock time for batch processing.

```python
pipe = _build_inpaint_pipe()  # load once
for img_path in all_images:
    phase3_lora_inpaint(img_path, pipe)  # reuse
```

### 9.7 LAB Colour Matching

After compositing, the feathered blend seam is further corrected by transferring the preserved base region's colour statistics to the generated region in LAB colourspace:

```python
def _color_match(generated_bgr, reference_bgr, mask_hw):
    gen_lab = bgr2lab(generated_bgr)
    ref_lab = bgr2lab(reference_bgr)
    ref_pixels = ref_lab[mask_hw < 128]   # unmasked = ground truth palette
    gen_region = mask_hw > 127            # generated region to correct
    for c in range(3):                    # L, A, B channels
        # Standardise generated region → rescale to reference statistics
        gen_lab[gen_region, c] = (
            (gen_lab[gen_region, c] - gen_mean[c]) / gen_std[c]
        ) * ref_std[c] + ref_mean[c]
    return lab2bgr(clip(gen_lab))
```

This corrects the primary visible defect in the previous results: the generated top had a noticeably different colour temperature (cooler/bluer) compared to the warm sandstone of the intact base.

### 9.8 Inference Configuration

```
num_inference_steps        = 50   (was 40)
guidance_scale             = 9.0  (was 8.5)
controlnet_conditioning_scale = 0.6
strength                   = 0.95
seed                       = 42
```

---

## Quantitative Results

| Stage | Key Metric | Value |
|---|---|---|
| Preprocessing | Sharpness improvement (Laplacian var) | **+177%** |
| Preprocessing | Exposure quality | Good on 11/12 images |
| SIFT | Mean keypoints per image | 4,123 |
| Matching | Connected image pairs | 47 / 66 possible (71%) |
| SfM | Registered camera poses | 20 |
| SfM | Sparse 3D points | 1,283 |
| Dense | Dense point cloud | 33,261 points |
| Inpainting | Structural SSIM vs original | 0.4006 |
| LoRA | Trainable parameters (rank 8) | ~12M (~1.4% of attention params) |
| LoRA | Weight file size | ~24 MB |
| LoRA | Training steps | 500 |
| LoRA | Images processed (batch) | 7 |
| Compute cost | All stages | $0 (free APIs + local) |

---

## Future Improvements

Ranked by expected impact on archaeological fidelity:

### 1. Larger LoRA Training Dataset ⭐⭐⭐⭐⭐

**Current:** 8 raw images → 96 augmented crops. This is enough for the model to learn Hampi's stone colour and approximate architectural character, but not enough to learn tier proportions, cornice profiles, or specific carving patterns reliably.

**Proposed:** Collect 100–300 images from:
- ASI (Archaeological Survey of India) digital archives
- Wikimedia Commons systematic crawl (all Hampi-tagged images)
- Field photography at 0.5m resolution

With 300+ images, the LoRA will learn Vijayanagara-specific vocabulary: double-tier cornice profiles, kirtimukha density, sandstone oxidation colour, entrance pillar proportions.

### 2. Paired Damage→Complete Dataset ⭐⭐⭐⭐⭐

Train on **image pairs**: (partially damaged gopuram, same gopuram with top complete) so the model learns the reconstruction task directly, not just architectural style.

Data sources:
- ASI before/after conservation photographs
- Historical colonial-era survey photos (c.1850–1920)
- **Synthetic pairs:** programmatically damage intact monument images to create cheap supervision

### 3. Depth + Surface Normal Conditioning ⭐⭐⭐⭐

Replace Canny edges with richer 3D-aware signals:
- **MiDaS / DepthAnything-v2:** per-pixel relative depth
- **NormalBae:** surface normals encoding relief curvature
- **MLSD line segments:** clean architectural geometry

These conditioning signals give the inpainting model a geometric prior aligned with the surviving 3D structure from the SfM stage.

### 4. IP-Adapter Reference Conditioning ⭐⭐⭐⭐

Pass a photograph of an **intact gopuram** (e.g. Virupaksha Temple tower, same Hampi complex) as a visual style anchor. IP-Adapter injects the reference image's CLIP embedding directly into the cross-attention:

```
Attention(Q,  K_text + K_image,  V_text + V_image)
```

The generated image inherits the reference's stone colour, carving rhythm, and tower profile more faithfully than any text prompt.

### 5. Pixel-Accurate Damage Segmentation ⭐⭐⭐

Replace the horizontal-line mask with a fine-tuned SAM or U-Net that produces a **per-pixel damage probability map**. The current Laplacian boundary over-inpaints partially intact cornices and under-inpaints laterally-damaged zones.

### 6. Multi-View NeRF Inpainting ⭐⭐⭐

Use the SfM point cloud as a 3D constraint: fill the missing volume in 3D space (SPIn-NeRF or similar) rather than in a single 2D image. The result is geometrically consistent across all viewing angles.

---

## Implementation Roadmap

| Priority | Recommendation | Effort | Cost | Timeline |
|---|---|---|---|---|
| 1 | Expanded LoRA dataset (100→300 images) | Medium | Free | 2–3 weeks |
| 2 | Synthetic paired damage dataset | High | Free | 4–6 weeks |
| 3 | Depth + normal conditioning | Low | Free | 1 week |
| 4 | IP-Adapter reference image | Low | Free | 3–5 days |
| 5 | SAM damage segmentation | Medium | Free | 2–3 weeks |
| 6 | Multi-view NeRF inpainting | High | Free/Low | 6–8 weeks |

**Fastest high-impact path:** Depth conditioning (1 week) → IP-Adapter (1 week) → expanded LoRA dataset (3 weeks).

---

## Broader Vision

The combination of these improvements would move the project from "demonstrates the concept" to **archaeologically useful** — a restoration tool that ASI conservators and heritage researchers could use to visualise and document missing structural elements. The long-term ambition is a pipeline that ingests any photograph of a partially destroyed Hampi monument and produces a publication-quality architectural restoration, grounded in the specific Vijayanagara building canon and validated against iconographic records.

This would be among the first computational tools purpose-built for Deccan medieval architecture, filling a gap that no commercial or academic tool currently addresses.

---

## Running the Pipeline

```bash
# Setup
git clone <repo>
cd hampi-revived
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Full 3D reconstruction pipeline
python pipeline.py

# SDXL baseline inpainting
python generative_reconstruct.py

# End-to-end LoRA pipeline (train + inpaint + results)
python lora_pipeline.py

# Streamlit web app
streamlit run app.py

# Optional: target a single image (default = all 7 raw images)
IMG_PATH=data/raw/7f9fc5ee81.jpg python lora_pipeline.py

# Optional: force retrain LoRA (required after changing rank/steps)
FORCE_RETRAIN=1 python lora_pipeline.py

# Optional: disable ControlNet for faster (lower quality) inference
USE_CONTROLNET=0 python lora_pipeline.py

# Optional: override training steps
TRAIN_STEPS=800 python lora_pipeline.py
```
