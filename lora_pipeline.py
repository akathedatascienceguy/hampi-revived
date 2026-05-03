"""
End-to-end LoRA pipeline for Hampi Gopuram restoration.

Phase 1 — Data prep     : augment raw Hampi images → training set
Phase 2 — LoRA training : fine-tune SD-2.1-base UNet attention on Hampi imagery
Phase 3 — LoRA inpaint  : restore damaged gopurams with the trained LoRA
Phase 4 — Results       : comparison figures saved to outputs/lora_<stem>.png

Run:
    source venv/bin/activate
    python lora_pipeline.py                          # all images
    IMG_PATH=data/raw/59b2b09ec5.jpg python lora_pipeline.py  # single image

Optional env-vars:
    IMG_PATH=data/raw/59b2b09ec5.jpg   (single target; default = all raw images)
    TRAIN_STEPS=80                      (default 80)
    FORCE_RETRAIN=1                     (ignore cached LoRA weights)
"""

import os, sys, warnings, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import preprocess_image

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
RAW_DIR    = ROOT / "data" / "raw"
TRAIN_DIR  = ROOT / "data" / "lora_train"
LORA_DIR   = ROOT / "outputs" / "lora_weights"
OUT_PATH   = ROOT / "outputs" / "lora_reconstruction.png"
for d in [TRAIN_DIR, LORA_DIR, ROOT / "outputs"]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────────────────
_IMG_PATH_ENV = os.environ.get("IMG_PATH", "")
IMG_PATH      = _IMG_PATH_ENV if _IMG_PATH_ENV else None   # None = batch all raw images
BASE_MODEL    = "Lykon/dreamshaper-8"          # SD-1.5 fine-tune, freely accessible
INPAINT_MODEL = "Lykon/dreamshaper-8-inpainting"  # matched inpainting variant
LORA_FILE     = LORA_DIR / "hampi_lora.pt"
LORA_RANK     = 4
LORA_ALPHA    = 16.0
TRAIN_STEPS   = int(os.environ.get("TRAIN_STEPS", 80))   # 80 steps ≈ 30–40 min on MPS
FORCE_RETRAIN = os.environ.get("FORCE_RETRAIN", "0") == "1"
BATCH_SIZE    = 1
LR            = 1e-4
IMG_SIZE      = 512

TRAIN_CAPTION = (
    "ancient Hampi Vijayanagara carved stone temple ruins, granite gopuram, "
    "ornate carved pillars and friezes, historical India, photorealistic"
)
INPAINT_PROMPT = (
    "ancient Hampi Vijayanagara stone gopuram, complete intact shikhara tower "
    "rising above the ornate carved entrance arch, all stone tiers fully intact, "
    "matching granite sandstone texture, photorealistic"
)
NEG_PROMPT = (
    "modern, blurry, people, trees growing from structure, low quality, "
    "damaged, broken, cartoon, painting, watermark"
)

# ─── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[device] Apple MPS (Metal Performance Shaders)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[device] CUDA — {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("[device] CPU (training will be slow)")

DTYPE = torch.float32  # MPS requires float32; CUDA can use bfloat16


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Augmentation
# ══════════════════════════════════════════════════════════════════════════════
def phase1_augment() -> int:
    print("\n" + "═" * 66)
    print("PHASE 1 — Data Augmentation")
    print("═" * 66)

    def clahe_enhance(img: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = cv2.merge([clahe.apply(l), a, b])
        rgb = cv2.cvtColor(cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    raw_imgs = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))
    print(f"  {len(raw_imgs)} raw images found in {RAW_DIR}")

    # Clear previous training crops
    for f in TRAIN_DIR.glob("*"):
        f.unlink()

    aug_count = 0
    for img_path in raw_imgs:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        sq = min(w, h)

        def sq_crop(im: Image.Image, ox: int = 0, oy: int = 0) -> Image.Image:
            return im.crop((ox, oy, ox + sq, oy + sq)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        base = sq_crop(img, (w - sq) // 2, (h - sq) // 2)

        variants: list[Image.Image] = [
            base,
            base.transpose(Image.FLIP_LEFT_RIGHT),
            clahe_enhance(base),
            clahe_enhance(base).transpose(Image.FLIP_LEFT_RIGHT),
            ImageEnhance.Brightness(base).enhance(0.85),
            ImageEnhance.Brightness(base).enhance(1.15),
            ImageEnhance.Contrast(base).enhance(1.2),
            ImageEnhance.Sharpness(base).enhance(1.5),
        ]

        # 2 random off-centre crops
        rng = np.random.default_rng(42)
        for _ in range(2):
            ox = int(rng.integers(0, max(1, w - sq)))
            oy = int(rng.integers(0, max(1, h - sq)))
            crop = sq_crop(img, ox, oy)
            variants.append(crop)
            variants.append(crop.transpose(Image.FLIP_LEFT_RIGHT))

        for v in variants:
            stem = f"hampi_{aug_count:04d}"
            v.save(TRAIN_DIR / f"{stem}.png")
            (TRAIN_DIR / f"{stem}.txt").write_text(TRAIN_CAPTION)
            aug_count += 1

    print(f"  {aug_count} augmented training images → {TRAIN_DIR}")
    return aug_count


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — LoRA Training
# ══════════════════════════════════════════════════════════════════════════════
class HampiDataset(Dataset):
    def __init__(self, data_dir: Path, tokenizer):
        self.paths = sorted(data_dir.glob("*.png"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        caption_file = self.paths[idx].with_suffix(".txt")
        caption = caption_file.read_text().strip() if caption_file.exists() else TRAIN_CAPTION
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.transform(img),
            "input_ids": tokens.input_ids.squeeze(0),
        }


def phase2_train_lora() -> list[float]:
    print("\n" + "═" * 66)
    print("PHASE 2 — LoRA Training")
    print("═" * 66)

    if LORA_FILE.exists() and not FORCE_RETRAIN:
        print(f"  Cached LoRA weights found at {LORA_FILE}")
        print("  Set FORCE_RETRAIN=1 to retrain from scratch")
        return []  # no loss history — weights were pre-trained

    from diffusers import StableDiffusionPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    print(f"  Loading {BASE_MODEL} (download ~2 GB first time) …")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False
    )
    tokenizer     = pipe.tokenizer
    # VAE + text encoder: frozen, run on CPU to save MPS memory for UNet training
    text_encoder  = pipe.text_encoder.cpu()
    vae           = pipe.vae.cpu()
    unet          = pipe.unet.to(DEVICE)
    noise_sched   = DDPMScheduler.from_pretrained(BASE_MODEL, subfolder="scheduler")
    del pipe  # free pipeline object; keep individual components

    # Freeze everything
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # Inject LoRA into UNet cross-attention projections
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Gradient checkpointing reduces peak activation memory on MPS
    unet.enable_gradient_checkpointing()

    dataset = HampiDataset(TRAIN_DIR, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()), lr=LR
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    unet.train()
    losses: list[float] = []
    step = 0

    print(f"  Training {TRAIN_STEPS} steps  |  rank={LORA_RANK}  α={LORA_ALPHA}  lr={LR}")
    pbar = tqdm(total=TRAIN_STEPS, desc="  LoRA", ncols=80)

    while step < TRAIN_STEPS:
        for batch in loader:
            if step >= TRAIN_STEPS:
                break

            pv  = batch["pixel_values"]          # keep on CPU for VAE
            ids = batch["input_ids"]               # keep on CPU for text_encoder

            with torch.no_grad():
                # Encode on CPU (frozen, no MPS memory cost)
                latents = vae.encode(pv.to(vae.device, dtype=DTYPE)).latent_dist.sample()
                latents = (latents * vae.config.scaling_factor).to(DEVICE)
                enc_hs  = text_encoder(ids.to(text_encoder.device))[0].to(DEVICE)

            noise     = torch.randn_like(latents)
            bsz       = latents.shape[0]
            timesteps = torch.randint(
                0, noise_sched.config.num_train_timesteps, (bsz,)
            ).long().to(DEVICE)
            noisy_lat = noise_sched.add_noise(latents, noise, timesteps)

            noise_pred = unet(noisy_lat, timesteps, enc_hs).sample
            loss       = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            # Flush MPS memory after each step to prevent fragmentation
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            losses.append(loss.item())
            step += 1
            pbar.update(1)

            if step % 10 == 0:
                avg = np.mean(losses[-10:])
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.1e}")

    pbar.close()
    avg_final = np.mean(losses[-20:]) if len(losses) >= 20 else np.mean(losses)
    print(f"  Training complete — final loss: {avg_final:.4f}")

    # Save only LoRA delta weights (tiny vs full model)
    lora_state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
    torch.save(lora_state, LORA_FILE)
    size_mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / 1e6
    n_params = sum(v.numel() for v in lora_state.values())
    print(f"  LoRA saved → {LORA_FILE}")
    print(f"  LoRA: {n_params/1e6:.2f}M params  |  {size_mb:.1f} MB on disk")

    return losses


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — LoRA Inpainting
# ══════════════════════════════════════════════════════════════════════════════
def _detect_boundary(img_bgr: np.ndarray) -> int:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    oh      = img_bgr.shape[0]
    win     = 30
    lap     = cv2.Laplacian(gray, cv2.CV_64F)
    row_var = np.array([np.abs(lap[r:r + win]).var() for r in range(0, oh - win, win)])
    top_h   = row_var[: len(row_var) // 2]
    drop_i  = int(np.argmin(np.diff(top_h))) if len(top_h) > 1 else 0
    return int(np.clip((drop_i + 1) * win, oh * 0.15, oh * 0.50))


def _feathered_composite(
    top_bgr: np.ndarray, base_bgr: np.ndarray, boundary: int, band: int = 80
) -> np.ndarray:
    oh = base_bgr.shape[0]
    hb = band // 2
    te = min(boundary, oh - 10)
    alpha = np.zeros(oh, dtype=np.float32)
    alpha[: max(0, te - hb)] = 1.0
    for r in range(max(0, te - hb), min(oh, te + hb)):
        t = (r - (te - hb)) / band
        alpha[r] = 0.5 * (1.0 + np.cos(np.pi * t))
    a = alpha[:, None, None]
    return (top_bgr * a + base_bgr * (1.0 - a)).astype(np.uint8)


def phase3_lora_inpaint(img_path: str = None):
    img_path = img_path or IMG_PATH or str(RAW_DIR / "59b2b09ec5.jpg")
    print("\n" + "═" * 66)
    print(f"PHASE 3 — LoRA Inpainting  [{Path(img_path).name}]")
    print("═" * 66)

    from diffusers import StableDiffusionInpaintPipeline
    from peft import LoraConfig, get_peft_model

    # ── Image preparation ───────────────────────────────────────────────────
    raw_bgr  = cv2.imread(img_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    proc_bgr = preprocess_image(raw_bgr, denoise_img=True)
    oh, ow   = proc_bgr.shape[:2]

    W = H = IMG_SIZE
    boundary    = _detect_boundary(proc_bgr)
    br_scaled   = int(boundary * H / oh)
    print(f"  Damage boundary: row {boundary}/{oh}  →  scaled {br_scaled}/{H}")

    orig_pil = Image.fromarray(
        cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    ).resize((W, H), Image.LANCZOS)

    mask_arr = np.zeros((H, W), dtype=np.uint8)
    mask_arr[:br_scaled, :] = 255
    mask_pil = Image.fromarray(mask_arr)

    # ── Load inpainting pipeline ────────────────────────────────────────────
    print(f"  Loading {INPAINT_MODEL} …")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL, torch_dtype=DTYPE,
        safety_checker=None, requires_safety_checker=False
    )
    pipe.enable_attention_slicing()

    # ── Apply LoRA to inpainting UNet ──────────────────────────────────────
    print(f"  Injecting LoRA (rank={LORA_RANK}) into inpainting UNet …")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    if not LORA_FILE.exists():
        raise FileNotFoundError(
            f"LoRA weights not found at {LORA_FILE}. "
            "Phase 2 must complete before Phase 3."
        )
    lora_state = torch.load(LORA_FILE, map_location="cpu")
    missing, unexpected = pipe.unet.load_state_dict(lora_state, strict=False)
    loaded = len(lora_state) - len(missing)
    print(f"  Loaded {loaded}/{len(lora_state)} LoRA tensors  |  unexpected: {len(unexpected)}")
    pipe.unet.eval()

    pipe = pipe.to(DEVICE)

    # ── Run inpainting ──────────────────────────────────────────────────────
    print("  Running LoRA-conditioned inpainting (40 steps) …")
    gen = torch.Generator(device=DEVICE).manual_seed(42)
    with torch.inference_mode():
        result = pipe(
            prompt=INPAINT_PROMPT,
            negative_prompt=NEG_PROMPT,
            image=orig_pil,
            mask_image=mask_pil,
            num_inference_steps=40,
            guidance_scale=8.5,
            strength=0.95,
            generator=gen,
        )
    inpainted_pil = result.images[0]
    print("  Inpainting done")

    # ── Feathered composite ─────────────────────────────────────────────────
    inpaint_bgr = cv2.cvtColor(
        np.array(inpainted_pil.resize((ow, oh), Image.LANCZOS)), cv2.COLOR_RGB2BGR
    )
    composite_bgr = _feathered_composite(inpaint_bgr, proc_bgr, boundary)

    return raw_bgr, proc_bgr, orig_pil, mask_pil, inpainted_pil, composite_bgr, boundary


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Results Figure
# ══════════════════════════════════════════════════════════════════════════════
def phase4_visualise(
    raw_bgr, proc_bgr, orig_pil, mask_pil,
    inpainted_pil, composite_bgr, boundary, losses,
    out_path: Path = None,
):
    print("\n" + "═" * 66)
    print("PHASE 4 — Results Figure")
    print("═" * 66)

    def to_rgb(x):
        if isinstance(x, np.ndarray):
            return cv2.cvtColor(x, cv2.COLOR_BGR2RGB) if x.ndim == 3 else x
        return np.array(x)

    oh, ow = proc_bgr.shape[:2]

    # Change heatmap
    inpaint_bgr = cv2.cvtColor(
        np.array(inpainted_pil.resize((ow, oh), Image.LANCZOS)), cv2.COLOR_RGB2BGR
    )
    diff     = cv2.absdiff(proc_bgr, composite_bgr).astype(np.float32)
    dmax     = diff.mean(axis=2).max()
    diff_u8  = (diff.mean(axis=2) / dmax * 255).astype(np.uint8) if dmax > 0 else np.zeros((oh, ow), np.uint8)
    heatmap  = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)

    # Mask overlay for visualization
    vis_arr = np.array(orig_pil).copy()
    vis_arr[:int(boundary * IMG_SIZE / oh), :, 0] = np.clip(
        vis_arr[:int(boundary * IMG_SIZE / oh), :, 0] * 0.3 + 170, 0, 255
    ).astype(np.uint8)
    vis_arr[:int(boundary * IMG_SIZE / oh), :, 1:] = (
        vis_arr[:int(boundary * IMG_SIZE / oh), :, 1:] * 0.3
    ).astype(np.uint8)
    mask_vis_pil = Image.fromarray(vis_arr)

    has_losses = len(losses) > 0

    # Layout: 2 rows × 4 cols when we have a loss plot, else 2 × 3
    if has_losses:
        fig = plt.figure(figsize=(24, 14))
        fig.patch.set_facecolor("#0d0d0d")
        gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.12)
        img_axes = [
            fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]
        ax_loss = fig.add_subplot(gs[1, 3])
    else:
        fig = plt.figure(figsize=(21, 13))
        fig.patch.set_facecolor("#0d0d0d")
        gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.12)
        img_axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        ax_loss = None

    n_train = len(list(TRAIN_DIR.glob("*.png")))
    panels = [
        (to_rgb(raw_bgr),        "Original (Damaged)",        "Input — truncated/eroded gopuram top"),
        (to_rgb(proc_bgr),       "Enhanced Input",             "CLAHE + denoised + sharpened"),
        (np.array(mask_vis_pil), "Inpaint Mask Overlay",       "Red = fill region · intact base preserved"),
        (np.array(mask_pil),     "Binary Mask",                "White = inpaint · Black = keep"),
        (np.array(inpainted_pil),"LoRA Inpainting Output",     f"SD-2.1 + Hampi LoRA (rank {LORA_RANK}, {n_train} train imgs)"),
        (to_rgb(composite_bgr),  "Final LoRA Restoration",     "Cosine-blended: LoRA top + original base"),
        (to_rgb(heatmap),        "Change Heatmap",             "Bright = modified · Dark = unchanged"),
    ]

    for ax, (img, title, sub) in zip(img_axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=10.5, fontweight="bold", pad=5)
        ax.set_xlabel(sub, color="#999", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # Row labels
    img_axes[0].set_ylabel("INPUT  /  MASK", color="#66aaff", fontsize=10, fontweight="bold", labelpad=8)
    img_axes[4].set_ylabel("LORA  RESTORATION", color="#ff9955", fontsize=10, fontweight="bold", labelpad=8)

    # Training loss plot
    if has_losses and ax_loss is not None:
        window   = min(30, max(1, len(losses) // 10))
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax_loss.plot(losses, color="#555", alpha=0.35, lw=0.7, label="raw")
        ax_loss.plot(
            range(window - 1, len(losses)), smoothed,
            color="#ff6b35", lw=2.0, label=f"EMA-{window}"
        )
        ax_loss.set_facecolor("#1a1a1a")
        ax_loss.set_title("LoRA Training Loss", color="white", fontsize=10.5, fontweight="bold", pad=5)
        ax_loss.set_xlabel("Step", color="#888", fontsize=8)
        ax_loss.set_ylabel("Diffusion MSE", color="#888", fontsize=8)
        ax_loss.tick_params(colors="#555", labelsize=7)
        for sp in ax_loss.spines.values():
            sp.set_edgecolor("#333")
        ax_loss.legend(fontsize=7, labelcolor="#aaa", facecolor="#111", edgecolor="#333")
        final_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
        ax_loss.text(
            0.97, 0.93, f"final Δ = {final_loss:.4f}",
            transform=ax_loss.transAxes, color="#ff6b35",
            ha="right", va="top", fontsize=8.5,
        )

    n_aug = len(list(TRAIN_DIR.glob("*.png")))
    fig.suptitle(
        f"LoRA-Conditioned Inpainting — Hampi Gopuram Restoration\n"
        f"{INPAINT_MODEL}  ·  LoRA rank {LORA_RANK}  ·  "
        f"trained {TRAIN_STEPS} steps on {n_aug} augmented Hampi images",
        color="white", fontsize=12.5, fontweight="bold", y=0.99,
    )

    save_to = out_path or OUT_PATH
    plt.savefig(save_to, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved → {save_to}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    t0 = time.time()

    phase1_augment()
    losses = phase2_train_lora()

    # Determine which images to process
    if IMG_PATH:
        target_images = [IMG_PATH]
    else:
        target_images = sorted(
            list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))
        )
        print(f"\n[batch] Processing {len(target_images)} images from {RAW_DIR}")

    saved = []
    for i, img_path in enumerate(target_images, 1):
        img_path = str(img_path)
        stem = Path(img_path).stem
        out_path = ROOT / "outputs" / f"lora_{stem}.png"
        print(f"\n[{i}/{len(target_images)}] {Path(img_path).name}")
        outputs = phase3_lora_inpaint(img_path)
        phase4_visualise(*outputs, losses, out_path=out_path)
        saved.append(out_path)

    elapsed = time.time() - t0
    print(f"\n{'═'*66}")
    print(f"Done in {elapsed/60:.1f} min")
    for p in saved:
        print(f"  → {p}")
    print("═" * 66)
