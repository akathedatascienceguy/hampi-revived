"""
End-to-end LoRA pipeline for Hampi Gopuram restoration.

Phase 1 — Data prep       : augment raw + collected images → training set
Phase 2 — LoRA training   : synthetic damage-pair training on inpainting UNet
Phase 3 — LoRA inpaint    : ControlNet-Canny + LoRA + IP-Adapter reference
Phase 4 — Results         : comparison figures → outputs/lora_<stem>.png

Run:
    source venv/bin/activate
    python lora_pipeline.py                               # all raw images
    IMG_PATH=data/raw/59b2b09ec5.jpg python lora_pipeline.py

Optional env-vars:
    IMG_PATH=...          single target; default = all raw images
    TRAIN_STEPS=600       default 600
    FORCE_RETRAIN=1       ignore cached LoRA weights
    USE_CONTROLNET=0      disable ControlNet
    USE_IP_ADAPTER=0      disable IP-Adapter reference
    IP_ADAPTER_SCALE=0.45 strength of reference image conditioning (0-1)
"""

import os, sys, warnings, time, random
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from src.preprocessing import preprocess_image

# ─── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
RAW_DIR   = ROOT / "data" / "raw"
REF_DIR   = ROOT / "data" / "reference"   # intact gopuram shots for IP-Adapter
TRAIN_DIR = ROOT / "data" / "lora_train"
LORA_DIR  = ROOT / "outputs" / "lora_weights"
OUT_PATH  = ROOT / "outputs" / "lora_reconstruction.png"
for d in [TRAIN_DIR, LORA_DIR, ROOT / "outputs", REF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Config ────────────────────────────────────────────────────────────────────
_IMG_ENV       = os.environ.get("IMG_PATH", "")
IMG_PATH       = _IMG_ENV or None
INPAINT_MODEL  = "Lykon/dreamshaper-8-inpainting"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
USE_CONTROLNET = os.environ.get("USE_CONTROLNET", "1") == "1"
USE_IP_ADAPTER = os.environ.get("USE_IP_ADAPTER", "1") == "1"
IP_ADAPTER_SCALE = float(os.environ.get("IP_ADAPTER_SCALE", "0.45"))
IP_ADAPTER_WEIGHTS = "ip-adapter_sd15.bin"   # standard (not plus) — lower VRAM
LORA_FILE      = LORA_DIR / "hampi_lora.pt"
LORA_RANK      = 8
LORA_ALPHA     = 32.0
TRAIN_STEPS    = int(os.environ.get("TRAIN_STEPS", 600))
FORCE_RETRAIN  = os.environ.get("FORCE_RETRAIN", "0") == "1"
BATCH_SIZE     = 1
LR             = 1e-4
IMG_SIZE       = 512

TRAIN_CAPTION = (
    "ancient Hampi Vijayanagara carved stone temple ruins, granite gopuram, "
    "ornate carved pillars and friezes, historical India, photorealistic"
)
INPAINT_PROMPT = (
    "ancient Hampi Vijayanagara stone gopuram, complete intact shikhara tower "
    "rising above the ornate carved entrance arch, all tiers fully intact, "
    "matching granite sandstone texture, same stone colour, photorealistic"
)
NEG_PROMPT = (
    "modern, blurry, people, trees, low quality, damaged, broken, cartoon, "
    "watermark, different building, different temple, colourful paint"
)

# ─── Device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("[device] Apple MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"[device] CUDA — {torch.cuda.get_device_name()}")
else:
    DEVICE = torch.device("cpu")
    print("[device] CPU (slow)")

DTYPE = torch.float32


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Data Augmentation
# ══════════════════════════════════════════════════════════════════════════════
def phase1_augment() -> int:
    print("\n" + "═" * 66)
    print("PHASE 1 — Data Augmentation")
    print("═" * 66)

    def clahe(img: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        rgb = cv2.cvtColor(cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # Include reference images in training set (intact gopurams from other temples)
    raw_imgs = sorted(
        list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")) +
        list(REF_DIR.glob("*.jpg")) + list(REF_DIR.glob("*.png"))
    )
    print(f"  {len(raw_imgs)} source images (raw + reference)")

    for f in TRAIN_DIR.glob("*"):
        f.unlink()

    aug_count = 0
    for img_path in raw_imgs:
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        w, h = img.size
        sq = min(w, h)

        def sq_crop(im, ox=0, oy=0):
            return im.crop((ox, oy, ox + sq, oy + sq)).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        base = sq_crop(img, (w - sq) // 2, (h - sq) // 2)
        variants = [
            base,
            base.transpose(Image.FLIP_LEFT_RIGHT),
            clahe(base),
            clahe(base).transpose(Image.FLIP_LEFT_RIGHT),
            ImageEnhance.Brightness(base).enhance(0.85),
            ImageEnhance.Brightness(base).enhance(1.15),
            ImageEnhance.Contrast(base).enhance(1.2),
            ImageEnhance.Sharpness(base).enhance(1.5),
        ]
        rng = np.random.default_rng(42)
        for _ in range(2):
            ox = int(rng.integers(0, max(1, w - sq)))
            oy = int(rng.integers(0, max(1, h - sq)))
            crop = sq_crop(img, ox, oy)
            variants += [crop, crop.transpose(Image.FLIP_LEFT_RIGHT)]

        for v in variants:
            stem = f"hampi_{aug_count:04d}"
            v.save(TRAIN_DIR / f"{stem}.png")
            (TRAIN_DIR / f"{stem}.txt").write_text(TRAIN_CAPTION)
            aug_count += 1

    print(f"  {aug_count} augmented training crops → {TRAIN_DIR}")
    return aug_count


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Synthetic Damage Pair LoRA Training
# ══════════════════════════════════════════════════════════════════════════════
class HampiDataset(Dataset):
    """
    For each intact training image, generates a random synthetic upper-region
    mask (15–65% from top).  The model sees the intact lower portion as context
    and must reconstruct the masked upper portion — directly matching the
    inference task of restoring damaged gopuram tops.
    """
    def __init__(self, data_dir: Path, tokenizer):
        self.paths = sorted(data_dir.glob("*.png"))
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.rng = random.Random(42)

    def __len__(self):
        return len(self.paths)

    def _make_mask(self) -> torch.Tensor:
        """Random upper-region binary mask in latent space (H/8 × W/8)."""
        lat_h = lat_w = IMG_SIZE // 8   # 64 × 64
        # Fraction of height to mask (top)
        frac = self.rng.uniform(0.15, 0.65)
        top  = max(1, int(frac * lat_h))
        mask = torch.zeros(1, lat_h, lat_w, dtype=torch.float32)
        mask[:, :top, :] = 1.0
        return mask

    def __getitem__(self, idx):
        img    = Image.open(self.paths[idx]).convert("RGB")
        cap_f  = self.paths[idx].with_suffix(".txt")
        caption = cap_f.read_text().strip() if cap_f.exists() else TRAIN_CAPTION
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.transform(img),
            "input_ids":    tokens.input_ids.squeeze(0),
            "mask":         self._make_mask(),   # (1, 64, 64)
        }


def phase2_train_lora() -> list[float]:
    print("\n" + "═" * 66)
    print("PHASE 2 — LoRA Training (synthetic damage pairs)")
    print("═" * 66)

    if LORA_FILE.exists() and not FORCE_RETRAIN:
        print(f"  Cached LoRA found at {LORA_FILE}")
        print("  Set FORCE_RETRAIN=1 to retrain")
        return []

    from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
    from peft import LoraConfig, get_peft_model

    print(f"  Loading {INPAINT_MODEL} …")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        INPAINT_MODEL, torch_dtype=DTYPE, safety_checker=None, requires_safety_checker=False
    )
    tokenizer    = pipe.tokenizer
    text_encoder = pipe.text_encoder.cpu()
    vae          = pipe.vae.cpu()
    unet         = pipe.unet.to(DEVICE)
    noise_sched  = DDPMScheduler.from_pretrained(INPAINT_MODEL, subfolder="scheduler")
    del pipe

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.05, bias="none",
    )
    unet = get_peft_model(unet, lora_cfg)
    unet.print_trainable_parameters()
    unet.enable_gradient_checkpointing()

    dataset = HampiDataset(TRAIN_DIR, tokenizer)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    optimizer    = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=LR)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS)

    unet.train()
    losses, step = [], 0
    print(f"  {TRAIN_STEPS} steps | rank={LORA_RANK} α={LORA_ALPHA} lr={LR}")
    pbar = tqdm(total=TRAIN_STEPS, desc="  LoRA", ncols=80)

    while step < TRAIN_STEPS:
        for batch in loader:
            if step >= TRAIN_STEPS:
                break

            pv  = batch["pixel_values"]        # (B, 3, 512, 512) CPU
            ids = batch["input_ids"]           # (B, 77)          CPU
            # Synthetic damage mask from dataset, shape (B, 1, 64, 64)
            mask_lat = batch["mask"].to(DEVICE, dtype=DTYPE)

            with torch.no_grad():
                latents = vae.encode(pv.to(vae.device, dtype=DTYPE)).latent_dist.sample()
                latents = (latents * vae.config.scaling_factor).to(DEVICE)
                enc_hs  = text_encoder(ids.to(text_encoder.device))[0].to(DEVICE)

                # masked_latents = intact latents with the damage zone zeroed out.
                # The model can see the preserved lower portion as context.
                masked_lat = latents * (1.0 - mask_lat)

            bsz       = latents.shape[0]
            noise     = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_sched.config.num_train_timesteps, (bsz,)
            ).long().to(DEVICE)
            noisy_lat = noise_sched.add_noise(latents, noise, timesteps)

            # 9-channel inpainting input: [noisy_latents | mask | masked_latents]
            unet_input = torch.cat([noisy_lat, mask_lat, masked_lat], dim=1)

            noise_pred = unet(unet_input, timesteps, enc_hs).sample
            # Only compute loss on the masked (damaged) region
            loss = F.mse_loss(
                (noise_pred * mask_lat).float(),
                (noise     * mask_lat).float(),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            if DEVICE.type == "mps":
                torch.mps.empty_cache()

            losses.append(loss.item())
            step += 1
            pbar.update(1)
            if step % 20 == 0:
                pbar.set_postfix(
                    loss=f"{np.mean(losses[-20:]):.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.1e}"
                )
            if step % 100 == 0:
                ckpt = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
                torch.save(ckpt, LORA_FILE)
                print(f"  [ckpt] step {step} — saved {LORA_FILE}")

    pbar.close()
    final_loss = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
    print(f"  Done — final loss: {final_loss:.4f}")

    lora_state = {k: v.cpu() for k, v in unet.state_dict().items() if "lora_" in k}
    torch.save(lora_state, LORA_FILE)
    mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / 1e6
    print(f"  Saved {LORA_FILE}  ({mb:.1f} MB)")
    return losses


# ══════════════════════════════════════════════════════════════════════════════
# MASK + COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _make_damage_mask(img_bgr: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Per-column sky detection: find where continuous sky from the top ends →
    that column's top-of-structure row.  Smooth, clamp, build per-pixel mask.
    """
    h, w = img_bgr.shape[:2]
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    sky  = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([85,  20,  80]), np.array([135, 255, 255])),
        cv2.inRange(hsv, np.array([0,    0, 180]), np.array([180,  35, 255])),
    )
    sky = cv2.dilate(sky, np.ones((7, 7), np.uint8), iterations=2)

    col_top = np.zeros(w, dtype=np.int32)
    for col in range(w):
        run = 0
        for r in range(h):
            if sky[r, col]:
                run = r + 1
            else:
                break
        col_top[col] = run

    col_top_f = np.convolve(col_top.astype(float), np.ones(61) / 61, mode="same")
    col_top_f = np.clip(col_top_f, h * 0.05, h * 0.65).astype(int)
    boundary  = int(np.median(col_top_f))

    mask = np.zeros((h, w), np.uint8)
    for col in range(w):
        top = col_top_f[col]
        if top > 0:
            mask[:top, col] = 255
    mask_f = cv2.GaussianBlur(mask.astype(np.float32), (51, 51), 0)
    return boundary, np.where(mask_f > 80, 255, 0).astype(np.uint8)


def _feathered_composite(top_bgr, base_bgr, boundary, band=80):
    oh = base_bgr.shape[0]
    hb = band // 2
    te = min(boundary, oh - 10)
    alpha = np.zeros(oh, np.float32)
    alpha[:max(0, te - hb)] = 1.0
    for r in range(max(0, te - hb), min(oh, te + hb)):
        t = (r - (te - hb)) / band
        alpha[r] = 0.5 * (1.0 + np.cos(np.pi * t))
    a = alpha[:, None, None]
    return (top_bgr * a + base_bgr * (1.0 - a)).astype(np.uint8)


def _color_match(generated_bgr, reference_bgr, mask_hw):
    """LAB colour statistics transfer: make generated region match preserved base."""
    gen_lab = cv2.cvtColor(generated_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference_bgr,  cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_px  = ref_lab[mask_hw < 128]
    gen_m   = mask_hw > 127
    if ref_px.shape[0] < 100:
        return generated_bgr
    result = gen_lab.copy()
    for c in range(3):
        rm, rs = ref_px[:, c].mean(), ref_px[:, c].std() + 1e-6
        src    = gen_lab[gen_m, c]
        sm, ss = src.mean(), src.std() + 1e-6
        result[gen_m, c] = (src - sm) / ss * rs + rm
    return cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# IP-ADAPTER REFERENCE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def _get_reference_image() -> Image.Image | None:
    """
    Pick the best reference image from data/reference/.
    Prefer landscape-oriented shots (wide gopuram views).
    Falls back to any image in reference/, then raw/.
    """
    candidates = sorted(
        list(REF_DIR.glob("*.jpg")) + list(REF_DIR.glob("*.png"))
    )
    if not candidates:
        # Fall back to raw images — pick the one with most intact structure
        candidates = sorted(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))

    if not candidates:
        return None

    # Score by aspect ratio close to portrait (tall=gopuram), and resolution
    def score(p):
        try:
            img = Image.open(p)
            w, h = img.size
            ratio = h / (w + 1e-6)
            return ratio * min(w, 1200)   # prefer tall, high-res
        except Exception:
            return 0.0

    best = max(candidates, key=score)
    try:
        ref = Image.open(best).convert("RGB")
        print(f"  IP-Adapter reference: {best.name}  {ref.size}")
        return ref.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    except Exception as e:
        print(f"  Reference load failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE BUILDER  (called once; reused across all images)
# ══════════════════════════════════════════════════════════════════════════════
def _build_inpaint_pipe():
    from peft import LoraConfig, get_peft_model

    if USE_CONTROLNET:
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        print(f"  Loading ControlNet ({CONTROLNET_MODEL}) …")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)
        print(f"  Loading {INPAINT_MODEL} + ControlNet …")
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            INPAINT_MODEL, controlnet=controlnet, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
        )
    else:
        from diffusers import StableDiffusionInpaintPipeline
        print(f"  Loading {INPAINT_MODEL} …")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            INPAINT_MODEL, torch_dtype=DTYPE,
            safety_checker=None, requires_safety_checker=False,
        )

    pipe.enable_attention_slicing()

    # ── IP-Adapter ──────────────────────────────────────────────────────────
    if USE_IP_ADAPTER:
        try:
            print(f"  Loading IP-Adapter ({IP_ADAPTER_WEIGHTS}) …")
            pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models",
                weight_name=IP_ADAPTER_WEIGHTS,
            )
            pipe.set_ip_adapter_scale(IP_ADAPTER_SCALE)
            print(f"  IP-Adapter scale: {IP_ADAPTER_SCALE}")
        except Exception as e:
            print(f"  IP-Adapter load failed (will skip): {e}")

    # ── LoRA injection ──────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    pipe.unet = get_peft_model(pipe.unet, lora_cfg)

    if not LORA_FILE.exists():
        raise FileNotFoundError(f"LoRA weights not found at {LORA_FILE}. Run phase 2 first.")

    lora_state = torch.load(LORA_FILE, map_location="cpu")
    missing, unexpected = pipe.unet.load_state_dict(lora_state, strict=False)
    loaded = len([k for k in lora_state if k not in set(missing)])
    print(f"  LoRA: {loaded}/{len(lora_state)} tensors  |  unexpected: {len(unexpected)}")
    pipe.unet.eval()
    pipe = pipe.to(DEVICE)
    print("  Pipeline ready.\n")
    return pipe


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — LoRA + ControlNet + IP-Adapter Inpainting
# ══════════════════════════════════════════════════════════════════════════════
def phase3_lora_inpaint(img_path: str, pipe, reference_pil: Image.Image | None = None):
    print("\n" + "═" * 66)
    print(f"PHASE 3 — [{Path(img_path).name}]")
    print("═" * 66)

    raw_bgr  = cv2.imread(img_path)
    if raw_bgr is None:
        raise FileNotFoundError(f"Cannot read {img_path}")
    proc_bgr = preprocess_image(raw_bgr, denoise_img=True)
    oh, ow   = proc_bgr.shape[:2]
    W = H    = IMG_SIZE

    boundary, mask_full = _make_damage_mask(proc_bgr)
    print(f"  Damage boundary: row {boundary}/{oh}")

    orig_pil = Image.fromarray(
        cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    ).resize((W, H), Image.LANCZOS)

    mask_512 = cv2.resize(mask_full, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_pil = Image.fromarray(mask_512)

    # ── ControlNet: Canny on preserved lower region ─────────────────────────
    if USE_CONTROLNET:
        gray    = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2GRAY)
        canny   = cv2.Canny(gray, 80, 200)
        canny[mask_512 > 127] = 0   # free the model in the generation zone
        control_pil = Image.fromarray(np.stack([canny, canny, canny], axis=-1))
    else:
        control_pil = None

    # ── Inference ───────────────────────────────────────────────────────────
    flags = []
    if USE_CONTROLNET:  flags.append("ControlNet")
    if USE_IP_ADAPTER and reference_pil is not None: flags.append("IP-Adapter")
    flags.append("LoRA")
    print(f"  Running: {' + '.join(flags)} (50 steps) …")

    gen = torch.Generator(device=DEVICE).manual_seed(42)
    call_kwargs = dict(
        prompt=INPAINT_PROMPT,
        negative_prompt=NEG_PROMPT,
        image=orig_pil,
        mask_image=mask_pil,
        num_inference_steps=50,
        guidance_scale=9.0,
        strength=0.95,
        generator=gen,
    )
    if USE_CONTROLNET:
        call_kwargs["control_image"] = control_pil
        call_kwargs["controlnet_conditioning_scale"] = 0.55
    if USE_IP_ADAPTER and reference_pil is not None:
        call_kwargs["ip_adapter_image"] = reference_pil

    with torch.inference_mode():
        result = pipe(**call_kwargs)

    inpainted_pil = result.images[0]
    print("  Done")

    # ── Composite + colour match ─────────────────────────────────────────────
    inpaint_bgr   = cv2.cvtColor(
        np.array(inpainted_pil.resize((ow, oh), Image.LANCZOS)), cv2.COLOR_RGB2BGR
    )
    mask_full_rs  = cv2.resize(mask_full, (ow, oh), interpolation=cv2.INTER_NEAREST)
    composite_bgr = _feathered_composite(inpaint_bgr, proc_bgr, boundary)
    composite_bgr = _color_match(composite_bgr, proc_bgr, mask_full_rs)

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
    inpaint_bgr = cv2.cvtColor(
        np.array(inpainted_pil.resize((ow, oh), Image.LANCZOS)), cv2.COLOR_RGB2BGR
    )
    diff    = cv2.absdiff(proc_bgr, composite_bgr).astype(np.float32)
    dmax    = diff.mean(axis=2).max()
    diff_u8 = (diff.mean(axis=2) / dmax * 255).astype(np.uint8) if dmax > 0 else np.zeros((oh, ow), np.uint8)
    heatmap = cv2.applyColorMap(diff_u8, cv2.COLORMAP_INFERNO)

    vis = np.array(orig_pil).copy()
    br  = int(boundary * IMG_SIZE / oh)
    vis[:br, :, 0] = np.clip(vis[:br, :, 0] * 0.3 + 170, 0, 255).astype(np.uint8)
    vis[:br, :, 1:] = (vis[:br, :, 1:] * 0.3).astype(np.uint8)
    mask_vis = Image.fromarray(vis)

    has_losses = len(losses) > 0
    if has_losses:
        fig = plt.figure(figsize=(24, 14))
        fig.patch.set_facecolor("#0d0d0d")
        gs = fig.add_gridspec(2, 4, hspace=0.32, wspace=0.12)
        axes = [
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
        axes   = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
        ax_loss = None

    n_train = len(list(TRAIN_DIR.glob("*.png")))
    panels = [
        (to_rgb(raw_bgr),        "Original (Damaged)",         "Input — eroded/truncated gopuram"),
        (to_rgb(proc_bgr),       "Enhanced Input",              "CLAHE + denoise + sharpen"),
        (np.array(mask_vis),     "Damage Mask (sky-contour)",   "Per-column HSV sky boundary"),
        (np.array(mask_pil),     "Binary Mask",                 "White = inpaint · Black = keep"),
        (np.array(inpainted_pil),"LoRA + ControlNet + IP-Adapt",
         f"rank {LORA_RANK}  α={LORA_ALPHA:.0f}  {TRAIN_STEPS} steps  {n_train} crops"),
        (to_rgb(composite_bgr),  "Final Restoration",           "Feathered blend + LAB colour match"),
        (to_rgb(heatmap),        "Change Heatmap",              "Bright = modified · Dark = unchanged"),
    ]

    for ax, (img, title, sub) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, color="white", fontsize=10.5, fontweight="bold", pad=5)
        ax.set_xlabel(sub, color="#999", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    axes[0].set_ylabel("INPUT / MASK", color="#66aaff", fontsize=10, fontweight="bold", labelpad=8)
    axes[4].set_ylabel("RESTORATION", color="#ff9955", fontsize=10, fontweight="bold", labelpad=8)

    if has_losses and ax_loss is not None:
        win      = min(30, max(1, len(losses) // 10))
        smoothed = np.convolve(losses, np.ones(win) / win, mode="valid")
        ax_loss.plot(losses, color="#555", alpha=0.35, lw=0.7, label="raw")
        ax_loss.plot(range(win - 1, len(losses)), smoothed, color="#ff6b35", lw=2.0, label=f"EMA-{win}")
        ax_loss.set_facecolor("#1a1a1a")
        ax_loss.set_title("Training Loss (masked region)", color="white", fontsize=10, fontweight="bold", pad=5)
        ax_loss.set_xlabel("Step", color="#888", fontsize=8)
        ax_loss.set_ylabel("MSE (damage zone)", color="#888", fontsize=8)
        ax_loss.tick_params(colors="#555", labelsize=7)
        for sp in ax_loss.spines.values():
            sp.set_edgecolor("#333")
        ax_loss.legend(fontsize=7, labelcolor="#aaa", facecolor="#111", edgecolor="#333")
        fl = np.mean(losses[-50:]) if len(losses) >= 50 else np.mean(losses)
        ax_loss.text(0.97, 0.93, f"final Δ = {fl:.4f}", transform=ax_loss.transAxes,
                     color="#ff6b35", ha="right", va="top", fontsize=8.5)

    ip_tag = f"IP-Adapter {IP_ADAPTER_SCALE}" if USE_IP_ADAPTER else ""
    cn_tag = "ControlNet-Canny" if USE_CONTROLNET else ""
    tags   = " · ".join(t for t in [cn_tag, ip_tag, f"LoRA rank {LORA_RANK}"] if t)

    fig.suptitle(
        f"Hampi Gopuram Restoration — Synthetic Damage Pair Training\n"
        f"{INPAINT_MODEL}  ·  {tags}  ·  {TRAIN_STEPS} steps  ·  {n_train} crops",
        color="white", fontsize=12, fontweight="bold", y=0.99,
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

    print("\n" + "═" * 66)
    print("BUILDING INPAINT PIPELINE")
    print("═" * 66)
    pipe = _build_inpaint_pipe()

    reference_pil = _get_reference_image() if USE_IP_ADAPTER else None

    if IMG_PATH:
        target_images = [IMG_PATH]
    else:
        target_images = sorted(
            list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png"))
        )
        print(f"[batch] {len(target_images)} images from {RAW_DIR}")

    saved = []
    for i, img_path in enumerate(target_images, 1):
        img_path = str(img_path)
        stem     = Path(img_path).stem
        out_path = ROOT / "outputs" / f"lora_{stem}.png"
        print(f"\n[{i}/{len(target_images)}] {Path(img_path).name}")
        outputs  = phase3_lora_inpaint(img_path, pipe, reference_pil)
        phase4_visualise(*outputs, losses, out_path=out_path)
        saved.append(out_path)

    elapsed = time.time() - t0
    print(f"\n{'═'*66}")
    print(f"Done in {elapsed/60:.1f} min")
    for p in saved:
        print(f"  → {p}")
    print("═" * 66)
