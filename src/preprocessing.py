"""
preprocessing.py — Image preprocessing for 3D reconstruction.

Steps applied:
  1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
  2. Denoising (Non-local means)
  3. Sharpening (unsharp mask)
  4. Stone-texture enhancement (LAB colour adjustment)
  5. Quality assessment (blur metric, exposure)
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """CLAHE on luminance channel (preserves stone colour while boosting detail)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, h: float = 6.0) -> np.ndarray:
    """Fast non-local means denoising — reduces sensor noise without blurring edges."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def sharpen(img: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Unsharp mask to boost carved-stone edge detail."""
    blur = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1 + strength, blur, -strength, 0)


def estimate_blur(img: np.ndarray) -> float:
    """Laplacian variance — higher = sharper. Below ~80 is blurry."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def estimate_exposure(img: np.ndarray) -> Tuple[float, str]:
    """Mean brightness and qualitative label."""
    mean = float(img.mean())
    if mean < 60:
        label = "underexposed"
    elif mean > 200:
        label = "overexposed"
    else:
        label = "good"
    return mean, label


def preprocess_image(img: np.ndarray, denoise_img: bool = True) -> np.ndarray:
    """Full preprocessing stack for a single image."""
    img = enhance_contrast(img)
    if denoise_img:
        img = denoise(img)
    img = sharpen(img)
    return img


def preprocess_batch(
    images: List[np.ndarray],
    denoise_imgs: bool = True,
) -> Tuple[List[np.ndarray], List[dict]]:
    """
    Preprocess all images; return processed list + quality report per image.
    Low-quality images (very blurry / over-exposed) are flagged but kept.
    """
    processed, quality = [], []
    for i, img in enumerate(images):
        p = preprocess_image(img, denoise_imgs)
        blur = estimate_blur(p)
        brightness, exposure = estimate_exposure(p)
        q = {"idx": i, "blur_score": round(blur, 1), "brightness": round(brightness, 1), "exposure": exposure}
        if blur < 40:
            logger.warning(f"Image {i}: very low sharpness ({blur:.1f}) — may hurt SfM")
            q["warning"] = "low_sharpness"
        processed.append(p)
        quality.append(q)
        logger.debug(f"  Image {i}: blur={blur:.1f}, brightness={brightness:.1f} ({exposure})")
    return processed, quality
