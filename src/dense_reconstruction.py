"""
dense_reconstruction.py — Dense point cloud via stereo matching.

For each overlapping image pair that has a registered camera pose we:
  1. Compute a disparity map (Semi-Global Block Matching — SGBM)
  2. Back-project disparity to depth using camera intrinsics
  3. Project each pixel into 3D world coordinates
  4. Accumulate → dense coloured point cloud

This supplements the sparse SfM cloud with surface coverage.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Stereo matcher ───────────────────────────────────────────────────────────

def make_stereo_matcher(
    num_disparities: int = 64,
    block_size: int = 11,
) -> cv2.StereoSGBM:
    """SGBM — good balance of quality vs speed for textured stone surfaces."""
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


# ─── Depth estimation for a rectified stereo pair ─────────────────────────────

def compute_disparity(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Compute disparity map for a (pre-rectified) stereo pair.
    Returns disparity in pixels (float32, masked invalid < 0).
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    matcher = make_stereo_matcher()
    disp = matcher.compute(g1, g2).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    return disp


def disparity_to_depth(disp: np.ndarray, focal: float, baseline: float = 1.0) -> np.ndarray:
    """Z = f * B / d  (baseline B in scene units, typically normalised to 1)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(np.isfinite(disp) & (disp > 0), focal * baseline / disp, np.nan)
    return depth


# ─── Back-projection ──────────────────────────────────────────────────────────

def depth_to_points(
    depth: np.ndarray,
    img: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    depth_clip: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth map pixels into world 3D coordinates.

    Returns:
        pts3d  (N, 3) world-frame points
        colors (N, 3) float RGB
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0) & (depth < depth_clip)
    ys, xs = np.where(valid)

    # Camera-frame coordinates
    Z = depth[ys, xs]
    X = (xs - K[0, 2]) * Z / K[0, 0]
    Y = (ys - K[1, 2]) * Z / K[1, 1]
    pts_cam = np.stack([X, Y, Z], axis=1)  # (N, 3)

    # World frame: Xw = R^T (Xc - t)
    pts_world = (R.T @ (pts_cam.T - t)).T

    # Colours (RGB 0–1)
    bgr = img[ys, xs].astype(np.float32) / 255.0
    colors = bgr[:, ::-1]  # BGR → RGB

    return pts_world, colors


# ─── Main ─────────────────────────────────────────────────────────────────────

def dense_reconstruct(
    images: List[np.ndarray],
    camera_poses: List[dict],
    K: np.ndarray,
    focal_factor: float = 1.2,
    max_pairs: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run dense reconstruction on consecutive registered camera pairs.

    Returns:
        pts3d   (N, 3) dense point cloud
        colors  (N, 3) float RGB
    """
    h, w = images[0].shape[:2]
    focal = K[0, 0]
    pose_map = {p["idx"]: p for p in camera_poses}

    all_pts, all_cols = [], []
    pairs_done = 0

    registered_idxs = sorted(pose_map.keys())
    for a, b in zip(registered_idxs[:-1], registered_idxs[1:]):
        if pairs_done >= max_pairs:
            break
        if a >= len(images) or b >= len(images):
            continue

        logger.info(f"  Dense pair ({a},{b})…")
        disp = compute_disparity(images[a], images[b])
        depth = disparity_to_depth(disp, focal)

        pose = pose_map[a]
        pts, cols = depth_to_points(depth, images[a], K, pose["R"], pose["t"])
        if len(pts) > 100:
            all_pts.append(pts)
            all_cols.append(cols)
            logger.info(f"    → {len(pts):,} dense points")
        pairs_done += 1

    if not all_pts:
        logger.warning("Dense reconstruction yielded no points.")
        return np.zeros((0, 3)), np.zeros((0, 3))

    return np.vstack(all_pts), np.vstack(all_cols)


# ─── Voxel downsampling ───────────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Simple voxel grid downsampling (average within each voxel)."""
    if len(pts) == 0:
        return pts, cols
    indices = np.floor(pts / voxel_size).astype(np.int64)
    voxels: Dict[tuple, list] = {}
    for k, (idx, pt, col) in enumerate(zip(map(tuple, indices), pts, cols)):
        if idx not in voxels:
            voxels[idx] = ([], [])
        voxels[idx][0].append(pt)
        voxels[idx][1].append(col)
    pts_out = np.array([np.mean(v[0], axis=0) for v in voxels.values()])
    cols_out = np.array([np.mean(v[1], axis=0) for v in voxels.values()])
    return pts_out, cols_out
