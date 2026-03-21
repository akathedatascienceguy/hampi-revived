"""
data_ingestion.py — Download & load Hampi images from Wikimedia Commons.

If real images are unavailable offline, generates structured synthetic
stone-texture scenes that mimic carved-rock temple geometry so the rest
of the pipeline still runs end-to-end.
"""

import os
import time
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import requests
from PIL import Image, ImageFilter, ImageEnhance
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ─── Public-domain Wikimedia images of Hampi ──────────────────────────────────
HAMPI_WIKI_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Virupaksha_temple_hampi.jpg/1280px-Virupaksha_temple_hampi.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Stone_Chariot%2C_Hampi.jpg/1280px-Stone_Chariot%2C_Hampi.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Hampi_Vittala_Temple.jpg/1280px-Hampi_Vittala_Temple.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/10/Hampi_ruins_temple.jpg/1280px-Hampi_ruins_temple.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Hazara_Rama_Temple%2C_Hampi.jpg/1280px-Hazara_Rama_Temple%2C_Hampi.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Lotus_Mahal%2C_Hampi.jpg/1280px-Lotus_Mahal%2C_Hampi.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Hampi_Elephant_Stables.jpg/1280px-Hampi_Elephant_Stables.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Hampi-boulder.jpg/1280px-Hampi-boulder.jpg",
]


def download_images(
    save_dir: str = "data/raw",
    max_images: int = 8,
    target_size: Tuple[int, int] = (1024, 768),
) -> List[str]:
    """Download Hampi images from Wikimedia Commons."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    saved = []
    urls = HAMPI_WIKI_IMAGES[:max_images]
    headers = {"User-Agent": "HampiRevived/1.0 (archaeological-3d-reconstruction)"}

    for url in tqdm(urls, desc="Downloading Hampi images"):
        fname = hashlib.md5(url.encode()).hexdigest()[:8] + ".jpg"
        dest = os.path.join(save_dir, fname)
        if os.path.exists(dest):
            saved.append(dest)
            continue
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            img = Image.open(__import__("io").BytesIO(r.content)).convert("RGB")
            img = img.resize(target_size, Image.LANCZOS)
            img.save(dest, quality=92)
            saved.append(dest)
            time.sleep(0.4)  # be polite to Wikimedia
        except Exception as e:
            logger.warning(f"Could not download {url}: {e}")

    if not saved:
        logger.warning("No images downloaded — generating synthetic scenes.")
        saved = generate_synthetic_hampi_scenes(save_dir, n=max_images, size=target_size)

    logger.info(f"Dataset ready: {len(saved)} images in '{save_dir}'")
    return saved


def generate_synthetic_hampi_scenes(
    save_dir: str,
    n: int = 8,
    size: Tuple[int, int] = (1024, 768),
) -> List[str]:
    """
    Generate synthetic 'stone ruin' scenes:
      - Granite-textured background (Hampi's iconic pink granite)
      - Geometric primitives: stepped plinths, pillars, arches
    These are NOT real images but let the CV pipeline run end-to-end.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    saved = []
    W, H = size

    for i in range(n):
        # --- base granite texture ---
        noise = rng.integers(100, 170, (H, W, 3), dtype=np.uint8)
        noise[:, :, 0] = np.clip(noise[:, :, 0] + 30, 0, 255)  # reddish granite
        noise[:, :, 1] = np.clip(noise[:, :, 1] - 10, 0, 255)

        img_arr = noise.astype(np.float32)
        # Add smooth noise layers for stone texture
        for scale in [4, 8, 16, 32]:
            small = rng.integers(0, 60, (H // scale + 1, W // scale + 1), dtype=np.uint8)
            big = np.array(
                Image.fromarray(small).resize((W, H), Image.BILINEAR), dtype=np.float32
            )
            img_arr[:, :, 0] += big * 0.3
            img_arr[:, :, 1] += big * 0.25
            img_arr[:, :, 2] += big * 0.2
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr)

        # --- draw stepped plinth ---
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        cx = W // 2 + rng.integers(-80, 80)
        base_y = int(H * 0.75)
        for step in range(5):
            w = 300 - step * 40
            h = 30
            y = base_y - step * h
            col = tuple(int(c * (0.6 + step * 0.06)) for c in (160, 120, 80))
            draw.rectangle([cx - w // 2, y, cx + w // 2, y + h], fill=col)

        # --- draw pillars ---
        for px in [cx - 120, cx, cx + 120]:
            pillar_h = 180 + rng.integers(-20, 20)
            pw = 30
            py = base_y - 5 * 30 - pillar_h
            col = (130 + rng.integers(-10, 10), 95, 65)
            draw.rectangle([px - pw // 2, py, px + pw // 2, base_y - 5 * 30], fill=col)
            # capital
            draw.ellipse([px - pw, py - 20, px + pw, py + 10], fill=(150, 110, 80))

        # Slight blur → looks more photographic
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
        # Small perspective shift to simulate different viewpoints
        angle = rng.uniform(-3, 3)
        img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=(140, 110, 90))

        path = os.path.join(save_dir, f"synthetic_hampi_{i:02d}.jpg")
        img.save(path, quality=88)
        saved.append(path)

    return saved


def load_images(paths: List[str], target_size: Tuple[int, int] = (1024, 768)) -> List[np.ndarray]:
    """Load and resize images to numpy arrays (BGR for OpenCV)."""
    import cv2
    imgs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            logger.warning(f"Could not load {p}")
            continue
        img = cv2.resize(img, target_size)
        imgs.append(img)
    return imgs


def dataset_stats(images: List[np.ndarray]) -> dict:
    """Compute basic stats about the loaded dataset."""
    stats = {
        "n_images": len(images),
        "resolution": f"{images[0].shape[1]}×{images[0].shape[0]}" if images else "N/A",
        "mean_brightness": float(np.mean([img.mean() for img in images])),
        "std_brightness": float(np.std([img.mean() for img in images])),
    }
    return stats
