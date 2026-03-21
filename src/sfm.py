"""
sfm.py — Incremental Structure from Motion (SfM) pipeline.

For each pair of matched images:
  1. Extract matched point coordinates
  2. Estimate the Fundamental / Essential matrix (RANSAC)
  3. Recover camera pose (R, t)
  4. Triangulate 3D points
  5. Accumulate into a sparse point cloud + camera poses

Camera intrinsics: estimated from image dimensions (assumes ~60° FOV)
unless a calibration file is provided.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ─── Intrinsics ───────────────────────────────────────────────────────────────

def estimate_intrinsics(
    img_shape: Tuple[int, int],
    focal_factor: float = 1.2,
) -> np.ndarray:
    """
    Build a plausible camera matrix K from image dimensions.
    focal = focal_factor * max(W, H)
    Principal point = image centre.
    """
    h, w = img_shape[:2]
    f = focal_factor * max(w, h)
    K = np.array([
        [f, 0, w / 2],
        [0, f, h / 2],
        [0, 0,     1],
    ], dtype=np.float64)
    return K


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def pts_from_matches(
    kps1: List[cv2.KeyPoint],
    kps2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract matched (x, y) coordinates."""
    p1 = np.float32([kps1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kps2[m.trainIdx].pt for m in matches])
    return p1, p2


def estimate_pose(
    pts1: np.ndarray,
    pts2: np.ndarray,
    K: np.ndarray,
    ransac_threshold: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """
    Estimate relative camera pose using the essential matrix.
    Returns (R, t, mask) where mask marks inliers.
    Returns (None, None, mask) on failure.
    """
    E, mask = cv2.findEssentialMat(
        pts1, pts2, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold,
    )
    if E is None or mask is None:
        return None, None, np.array([])

    n_inliers = int(mask.sum())
    logger.debug(f"    Essential matrix inliers: {n_inliers}/{len(pts1)}")
    if n_inliers < 8:
        return None, None, mask

    _, R, t, mask2 = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    return R, t, (mask2 > 0).flatten()


def triangulate(
    pts1: np.ndarray,
    pts2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """Triangulate 3D points from two projection matrices and 2D correspondences."""
    pts1_h = pts1.T  # (2, N)
    pts2_h = pts2.T
    pts4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
    pts3d = pts4d[:3] / pts4d[3]  # homogeneous → Euclidean
    return pts3d.T  # (N, 3)


def filter_points(pts3d: np.ndarray, max_dist: float = 50.0) -> np.ndarray:
    """Remove points too far from scene centre (triangulation artefacts)."""
    centre = np.median(pts3d, axis=0)
    dists = np.linalg.norm(pts3d - centre, axis=1)
    mask = dists < max_dist
    return pts3d[mask]


# ─── Main SfM loop ────────────────────────────────────────────────────────────

class SfMPipeline:
    """Incremental SfM accumulator."""

    def __init__(self, K: np.ndarray, ransac_threshold: float = 1.0):
        self.K = K
        self.ransac_threshold = ransac_threshold
        self.points3d: List[np.ndarray] = []   # list of (N,3) arrays
        self.colors: List[np.ndarray] = []      # matching RGB per point
        self.camera_poses: List[dict] = []      # {idx, R, t, P}

        # First camera is world origin
        R0 = np.eye(3)
        t0 = np.zeros((3, 1))
        P0 = K @ np.hstack([R0, t0])
        self.camera_poses.append({"idx": 0, "R": R0, "t": t0, "P": P0})
        self._pose_map: Dict[int, dict] = {0: self.camera_poses[0]}

    def process_pair(
        self,
        i: int,
        j: int,
        kps_i: List[cv2.KeyPoint],
        kps_j: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        img_i: np.ndarray,
        img_j: np.ndarray,
    ) -> int:
        """Process one image pair → add triangulated points. Returns #new points."""
        pts_i, pts_j = pts_from_matches(kps_i, kps_j, matches)

        # Pose of image i (already registered or world origin)
        if i not in self._pose_map:
            logger.debug(f"    Image {i} not yet registered — skipping pair ({i},{j})")
            return 0
        pose_i = self._pose_map[i]

        R, t, mask = estimate_pose(pts_i, pts_j, self.K, self.ransac_threshold)
        if R is None:
            logger.warning(f"    Pose estimation failed for pair ({i},{j})")
            return 0

        # Compose pose j relative to world
        R_i = pose_i["R"]
        t_i = pose_i["t"]
        R_j = R @ R_i
        t_j = R @ t_i + t
        P_j = self.K @ np.hstack([R_j, t_j])

        if j not in self._pose_map:
            pose_j = {"idx": j, "R": R_j, "t": t_j, "P": P_j}
            self.camera_poses.append(pose_j)
            self._pose_map[j] = pose_j

        # Triangulate inlier correspondences
        inlier_pts_i = pts_i[mask]
        inlier_pts_j = pts_j[mask]
        if len(inlier_pts_i) < 4:
            return 0

        pts3d = triangulate(inlier_pts_i, inlier_pts_j, pose_i["P"], P_j)
        pts3d = filter_points(pts3d)

        # Sample colour from image i
        colors = sample_colors(pts3d, inlier_pts_i[:len(pts3d)], img_i)

        self.points3d.append(pts3d)
        self.colors.append(colors)
        logger.info(f"  Pair ({i},{j}): +{len(pts3d)} 3D points  ({len(self.camera_poses)} cameras registered)")
        return len(pts3d)

    def get_point_cloud(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return aggregated (N,3) points and (N,3) colours."""
        if not self.points3d:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pts = np.vstack(self.points3d)
        cols = np.vstack(self.colors)
        return pts, cols

    def camera_centres(self) -> np.ndarray:
        """World-frame camera centre for each registered pose."""
        centres = []
        for pose in self.camera_poses:
            C = -pose["R"].T @ pose["t"]
            centres.append(C.flatten())
        return np.array(centres)


def sample_colors(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    img: np.ndarray,
) -> np.ndarray:
    """
    Sample BGR pixel colours at projected 2D positions,
    return as float (0–1) RGB array.
    """
    h, w = img.shape[:2]
    n = min(len(pts3d), len(pts2d))
    colors = np.zeros((n, 3), dtype=np.float32)
    for k in range(n):
        x, y = int(pts2d[k, 0]), int(pts2d[k, 1])
        if 0 <= x < w and 0 <= y < h:
            b, g, r = img[y, x]
            colors[k] = [r / 255.0, g / 255.0, b / 255.0]
        else:
            colors[k] = [0.5, 0.4, 0.3]  # default granite tone
    return colors


def run_sfm(
    images: List[np.ndarray],
    all_kps: List[List[cv2.KeyPoint]],
    matches_dict: Dict[Tuple[int, int], List[cv2.DMatch]],
    img_shape: Tuple[int, int],
    focal_factor: float = 1.2,
    ransac_threshold: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
    """
    Full SfM run.

    Returns:
        points3d      (N, 3) sparse point cloud
        colors        (N, 3) float RGB
        camera_poses  list of pose dicts
        camera_centres (M, 3)
    """
    K = estimate_intrinsics(img_shape, focal_factor)
    logger.info(f"Camera matrix K:\n{K}")

    sfm = SfMPipeline(K, ransac_threshold)
    total_pts = 0

    for (i, j), matches in sorted(matches_dict.items()):
        n = sfm.process_pair(
            i, j,
            all_kps[i], all_kps[j],
            matches,
            images[i], images[j],
        )
        total_pts += n

    pts, cols = sfm.get_point_cloud()
    centres = sfm.camera_centres()
    logger.info(f"SfM complete: {len(pts)} 3D points, {len(sfm.camera_poses)} cameras")
    return pts, cols, sfm.camera_poses, centres
