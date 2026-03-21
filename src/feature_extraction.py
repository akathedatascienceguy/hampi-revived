"""
feature_extraction.py — SIFT feature detection, description & matching.

For Hampi's carved-granite architecture (rich in repeating texture and
distinctive silhouettes) SIFT is the gold standard: scale/rotation invariant,
handles the wide range of viewpoints encountered in field photography.

Output artefacts saved:
  outputs/features/keypoints_<i>.jpg
  outputs/features/matches_<i>_<j>.jpg
"""

import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ─── Detection ────────────────────────────────────────────────────────────────

def detect_and_describe(
    images: List[np.ndarray],
    detector: str = "SIFT",
    max_keypoints: int = 5000,
) -> Tuple[List[List[cv2.KeyPoint]], List[np.ndarray]]:
    """
    Detect keypoints and compute descriptors for each image.

    Returns:
        all_kps    : list of keypoint lists
        all_descs  : list of descriptor arrays (float32 for SIFT)
    """
    if detector == "SIFT":
        det = cv2.SIFT_create(nfeatures=max_keypoints)
    elif detector == "ORB":
        det = cv2.ORB_create(nfeatures=max_keypoints)
    elif detector == "AKAZE":
        det = cv2.AKAZE_create()
    else:
        raise ValueError(f"Unknown detector: {detector}")

    all_kps, all_descs = [], []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, descs = det.detectAndCompute(gray, None)
        all_kps.append(kps)
        all_descs.append(descs)
        logger.info(f"  Image {i}: {len(kps)} keypoints")
    return all_kps, all_descs


# ─── Matching ─────────────────────────────────────────────────────────────────

def match_pair(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio: float = 0.75,
    use_flann: bool = True,
) -> List[cv2.DMatch]:
    """
    Match descriptors between two images using Lowe's ratio test.
    FLANN for SIFT (fast); BFMatcher for binary descriptors (ORB/AKAZE).
    """
    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return []

    if use_flann and desc1.dtype == np.float32:
        index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
        search_params = {"checks": 50}
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    raw = matcher.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in raw if m.distance < ratio * n.distance]
    return good


def match_all_pairs(
    all_descs: List[np.ndarray],
    ratio: float = 0.75,
    min_matches: int = 20,
) -> Dict[Tuple[int, int], List[cv2.DMatch]]:
    """Match every pair (i, j) where i < j. Returns only pairs with ≥ min_matches."""
    pairs = {}
    for i, j in combinations(range(len(all_descs)), 2):
        matches = match_pair(all_descs[i], all_descs[j], ratio=ratio)
        if len(matches) >= min_matches:
            pairs[(i, j)] = matches
            logger.info(f"  Pair ({i},{j}): {len(matches)} good matches")
        else:
            logger.debug(f"  Pair ({i},{j}): {len(matches)} matches — too few, skipped")
    return pairs


# ─── Visualisation ────────────────────────────────────────────────────────────

def save_keypoints_plot(
    images: List[np.ndarray],
    all_kps: List[List[cv2.KeyPoint]],
    out_dir: str = "outputs/features",
    n_show: int = 4,
) -> List[str]:
    """Save keypoint overlay images (first n_show images)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(min(n_show, len(images))):
        vis = cv2.drawKeypoints(
            images[i],
            all_kps[i],
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        path = os.path.join(out_dir, f"keypoints_{i:02d}.jpg")
        cv2.imwrite(path, vis)
        paths.append(path)
    return paths


def save_matches_plot(
    images: List[np.ndarray],
    all_kps: List[List[cv2.KeyPoint]],
    matches_dict: Dict[Tuple[int, int], List[cv2.DMatch]],
    out_dir: str = "outputs/features",
    n_matches_draw: int = 50,
    max_pairs: int = 6,
) -> List[str]:
    """Save match visualisation for top pairs."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = []
    for (i, j), matches in list(matches_dict.items())[:max_pairs]:
        vis = cv2.drawMatches(
            images[i], all_kps[i],
            images[j], all_kps[j],
            matches[:n_matches_draw], None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        path = os.path.join(out_dir, f"matches_{i:02d}_{j:02d}.jpg")
        cv2.imwrite(path, vis)
        paths.append(path)
    return paths


def keypoint_stats(
    all_kps: List[List[cv2.KeyPoint]],
    matches_dict: Dict[Tuple[int, int], List[cv2.DMatch]],
) -> dict:
    """Summary statistics for a quick quality check."""
    kp_counts = [len(k) for k in all_kps]
    match_counts = [len(v) for v in matches_dict.values()]
    return {
        "total_images": len(all_kps),
        "keypoints": {
            "min": int(np.min(kp_counts)),
            "max": int(np.max(kp_counts)),
            "mean": float(np.mean(kp_counts)),
        },
        "connected_pairs": len(matches_dict),
        "matches": {
            "min": int(np.min(match_counts)) if match_counts else 0,
            "max": int(np.max(match_counts)) if match_counts else 0,
            "mean": float(np.mean(match_counts)) if match_counts else 0,
        },
    }


def plot_keypoint_distribution(
    all_kps: List[List[cv2.KeyPoint]],
    out_path: str = "outputs/features/keypoint_distribution.png",
) -> str:
    """Bar chart: #keypoints per image."""
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    counts = [len(k) for k in all_kps]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(counts)), counts, color="#c0763a", edgecolor="#7a4a1e", linewidth=0.8)
    ax.set_xlabel("Image Index", fontsize=12)
    ax.set_ylabel("Keypoints Detected", fontsize=12)
    ax.set_title("SIFT Keypoints per Image — Hampi Dataset", fontsize=14, fontweight="bold")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(c), ha="center", va="bottom", fontsize=9)
    ax.set_facecolor("#faf7f0")
    fig.patch.set_facecolor("#faf7f0")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
