"""
True inpainting pipeline for Hampi gopuram structure completion.
Uses diffusers/stable-diffusion-xl-inpainting (free HuggingFace Space).

Steps:
  1. Preprocess: CLAHE + denoise + sharpen
  2. Detect damage boundary via Laplacian variance
  3. Build inpainting mask (white = damaged top to fill)
  4. SDXL Inpainting via free HF Space — fills ONLY masked region
  5. Cosine-feathered composite: original intact base + inpainted tower
  6. Save 2×3 comparison figure
"""

import os, sys, shutil
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gradio_client import Client, handle_file

from src.preprocessing import preprocess_image

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_PATH = os.environ.get("IMG_PATH", "data/raw/59b2b09ec5.jpg")
OUT_PATH = "outputs/generative_reconstruction_inpaint.png"
CAND_DIR = Path("outputs/candidates")
CAND_DIR.mkdir(parents=True, exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

W, H = 768, 512    # working resolution for Space

PROMPT = (
    "ancient Hampi Vijayanagara stone gopuram, complete intact towering shikhara pyramid "
    "rising above the ornate carved entrance arch, all stone tiers fully intact, "
    "matching sandstone texture, photorealistic, 8K"
)
NEG = "blurry, modern, people, trees growing from structure, low quality, damaged, cartoon"

# ── Helpers ────────────────────────────────────────────────────────────────────
def cv2_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# ── Stage 1: Load & preprocess ────────────────────────────────────────────────
print(f"Loading {IMG_PATH} …")
raw_bgr  = cv2.imread(IMG_PATH)
proc_bgr = preprocess_image(raw_bgr, denoise_img=True)
oh, ow   = proc_bgr.shape[:2]

# ── Stage 2: Damage-boundary detection ────────────────────────────────────────
print("Detecting damage boundary …")
gray    = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2GRAY)
win     = 30
lap     = cv2.Laplacian(gray, cv2.CV_64F)
row_var = np.array([np.abs(lap[r:r+win]).var() for r in range(0, oh-win, win)])
top_half = row_var[:len(row_var)//2]
drop_idx = int(np.argmin(np.diff(top_half))) if len(top_half) > 1 else 0
boundary_row = int(np.clip((drop_idx + 1) * win, oh*0.15, oh*0.50))
br_scaled    = int(boundary_row * H / oh)   # scaled to working resolution
print(f"  Boundary at row {boundary_row}/{oh} (scaled → {br_scaled}/{H})")

# ── Stage 3: Build mask & prepare inputs ─────────────────────────────────────
print("Building inpainting mask …")
orig_pil = cv2_to_pil(proc_bgr).resize((W, H), Image.LANCZOS)

# White = fill (damaged top), black = keep (intact base)
mask_arr = np.zeros((H, W), dtype=np.uint8)
mask_arr[:br_scaled, :] = 255
mask_pil = Image.fromarray(mask_arr).convert("RGB")

# Mask overlay for visualisation
vis_arr = np.array(orig_pil).copy()
vis_arr[:br_scaled, :, 0] = np.clip(vis_arr[:br_scaled, :, 0]*0.4 + 180, 0, 255).astype(np.uint8)
vis_arr[:br_scaled, :, 1] = (vis_arr[:br_scaled, :, 1]*0.4).astype(np.uint8)
vis_arr[:br_scaled, :, 2] = (vis_arr[:br_scaled, :, 2]*0.4).astype(np.uint8)
cv2.putText(cv2.cvtColor(vis_arr, cv2.COLOR_RGB2BGR),
            f"INPAINT (damaged, {br_scaled}px)", (8, br_scaled-6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 80), 1)
cv2.putText(cv2.cvtColor(vis_arr, cv2.COLOR_RGB2BGR),
            "KEEP (intact)", (8, br_scaled+18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 80), 1)
mask_vis_pil = Image.fromarray(vis_arr)

tmp_orig = "/tmp/inpaint_orig.png"
tmp_mask = "/tmp/inpaint_mask.png"
orig_pil.save(tmp_orig)
mask_pil.save(tmp_mask)

# ── Stage 4: SDXL Inpainting via free HF Space ───────────────────────────────
print("\nConnecting to diffusers/stable-diffusion-xl-inpainting Space …")
client = Client("diffusers/stable-diffusion-xl-inpainting", verbose=False)
print("  Running inpainting (fills only the masked damaged region) …")

result = client.predict(
    input_image={
        "background": handle_file(tmp_orig),
        "layers":     [handle_file(tmp_mask)],
        "composite":  handle_file(tmp_orig),
    },
    prompt=PROMPT,
    negative_prompt=NEG,
    guidance_scale=8.0,
    steps=30,
    strength=0.99,
    scheduler="EulerDiscreteScheduler",
    api_name="/predict",
)
client.close()

# Space returns (original_path, inpainted_path)
paths = [r if isinstance(r, str) else r.get("path") for r in result if r is not None]
inpainted_pil = None
for i, p in enumerate(paths):
    if p:
        img = Image.open(p).convert("RGB")
        img.save(CAND_DIR / f"inpainted_{i}.png")
        # The inpainted result is the second item (slider after)
        if i == 1:
            inpainted_pil = img
            print(f"  Inpainted result saved → inpainted_1.png")

if inpainted_pil is None and paths:
    inpainted_pil = Image.open(paths[-1]).convert("RGB")
    print("  Using last result as inpainted output")

# ── Stage 5: Composite — original base + inpainted top ───────────────────────
print("\nCompositing: original intact base + inpainted tower …")
inpainted_pil_r = inpainted_pil.resize((ow, oh), Image.LANCZOS)
inpaint_bgr     = pil_to_cv2(inpainted_pil_r)

blend_band = 80
half_band  = blend_band // 2
top_end    = min(boundary_row, oh - 10)

alpha_1d = np.zeros(oh, dtype=np.float32)
alpha_1d[:max(0, top_end - half_band)] = 1.0
for r in range(max(0, top_end - half_band), min(oh, top_end + half_band)):
    t = (r - (top_end - half_band)) / blend_band
    alpha_1d[r] = 0.5 * (1.0 + np.cos(np.pi * t))   # cosine fade

alpha_2d  = alpha_1d[:, None, None]
composite = (inpaint_bgr * alpha_2d + proc_bgr * (1.0 - alpha_2d)).astype(np.uint8)

# ── Stage 6: Change heatmap ───────────────────────────────────────────────────
diff      = cv2.absdiff(proc_bgr, composite).astype(np.float32)
diff_max  = diff.mean(axis=2).max()
diff_norm = (diff.mean(axis=2) / diff_max * 255).astype(np.uint8) if diff_max > 0 else np.zeros_like(gray)
heatmap   = cv2.applyColorMap(diff_norm, cv2.COLORMAP_INFERNO)

# ── Stage 7: Figure ───────────────────────────────────────────────────────────
print("\nBuilding figure …")

def show(img):
    if isinstance(img, np.ndarray):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.array(img)

panels = [
    (show(raw_bgr),       "Original (Damaged)",        "Ruined/truncated top, intact carved base"),
    (np.array(mask_vis_pil), "Inpainting Mask",         "Red = fill region · Green label = keep region"),
    (np.array(inpainted_pil), "SDXL Inpainting Output", "Damaged top filled by model from context"),
    (show(composite),     "Final Composite Restoration","Inpainted tower + original intact base (cosine blend)"),
    (show(proc_bgr),      "Enhanced Input (reference)","CLAHE + denoised + sharpened"),
    (show(heatmap),       "Change Heatmap",             "Bright = pixels modified · Dark = unchanged original"),
]

fig, axes = plt.subplots(2, 3, figsize=(21, 13))
fig.patch.set_facecolor("#0e0e0e")

for ax, (img_data, title, subtitle) in zip(axes.flat, panels):
    ax.imshow(img_data)
    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=7)
    ax.set_xlabel(subtitle, color="#aaa", fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

axes[0, 0].set_ylabel("INPUT + MASK",              color="#66aaff", fontsize=10, fontweight="bold", labelpad=8)
axes[1, 0].set_ylabel("INPAINTING RECONSTRUCTION", color="#ff9955", fontsize=10, fontweight="bold", labelpad=8)

fig.suptitle(
    f"True Inpainting Structure Completion — Hampi Gopuram ({IMG_PATH.split('/')[-1]})\n"
    "Free · diffusers/stable-diffusion-xl-inpainting HF Space · Only damaged region filled",
    color="white", fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout(pad=1.8)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nDone → {OUT_PATH}")
