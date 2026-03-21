"""
visualization.py — Rich visualisations for every pipeline stage.

Outputs (all saved to outputs/visualizations/):
  • Image grid               — raw + preprocessed comparison
  • Feature match heatmap    — match count matrix across image pairs
  • Sparse 3D scatter        — Plotly interactive + Matplotlib static
  • Camera trajectory        — camera centres in 3D
  • Dense cloud slice        — orthographic top-view
  • Depth map montage        — side-by-side disparity maps
  • Pipeline summary figure  — one-page overview
"""

import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

logger = logging.getLogger(__name__)

OUT = "outputs/visualizations"


def _ensure(d: str = OUT) -> str:
    Path(d).mkdir(parents=True, exist_ok=True)
    return d


# ─── 1. Image grid ────────────────────────────────────────────────────────────

def plot_image_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 4,
    title: str = "Hampi Dataset",
    out_path: str = f"{OUT}/image_grid.png",
) -> str:
    _ensure(os.path.dirname(out_path))
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 2.8))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < n:
            img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(titles[i] if titles else f"#{i}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


# ─── 2. Match matrix ──────────────────────────────────────────────────────────

def plot_match_matrix(
    n_images: int,
    matches_dict: Dict[Tuple[int, int], list],
    out_path: str = f"{OUT}/match_matrix.png",
) -> str:
    _ensure(os.path.dirname(out_path))
    mat = np.zeros((n_images, n_images), dtype=int)
    for (i, j), m in matches_dict.items():
        mat[i, j] = mat[j, i] = len(m)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrBr", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="# matches")
    ax.set_xlabel("Image j")
    ax.set_ylabel("Image i")
    ax.set_title("Feature Match Count Matrix", fontsize=13, fontweight="bold")
    for i in range(n_images):
        for j in range(n_images):
            if mat[i, j] > 0:
                ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7, color="black")
    fig.patch.set_facecolor("#faf7f0")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


# ─── 3. Sparse 3D scatter (Matplotlib) ───────────────────────────────────────

def plot_sparse_cloud_3d(
    pts: np.ndarray,
    colors: np.ndarray,
    camera_centres: Optional[np.ndarray] = None,
    out_path: str = f"{OUT}/sparse_cloud_3d.png",
    max_pts: int = 8000,
) -> str:
    _ensure(os.path.dirname(out_path))
    if len(pts) == 0:
        logger.warning("No points to plot.")
        return ""

    # Subsample
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, colors = pts[idx], colors[idx]

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               c=colors, s=0.8, alpha=0.6, linewidths=0)

    if camera_centres is not None and len(camera_centres) > 0:
        ax.scatter(camera_centres[:, 0], camera_centres[:, 1], camera_centres[:, 2],
                   c="red", s=80, marker="^", label="Cameras", zorder=5, edgecolors="white")
        if len(camera_centres) > 1:
            ax.plot(camera_centres[:, 0], camera_centres[:, 1], camera_centres[:, 2],
                    "r--", linewidth=1.2, alpha=0.7)

    ax.set_xlabel("X"), ax.set_ylabel("Y"), ax.set_zlabel("Z")
    ax.set_title("Sparse 3D Point Cloud — Hampi SfM", fontsize=13, fontweight="bold")
    if camera_centres is not None:
        ax.legend(fontsize=10)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    for spine in ["x", "y", "z"]:
        ax.tick_params(axis=spine, colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


# ─── 4. Plotly interactive cloud ─────────────────────────────────────────────

def plot_sparse_cloud_plotly(
    pts: np.ndarray,
    colors: np.ndarray,
    camera_centres: Optional[np.ndarray] = None,
    out_path: str = f"{OUT}/sparse_cloud_interactive.html",
    max_pts: int = 10000,
) -> str:
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not available — skipping interactive plot.")
        return ""
    _ensure(os.path.dirname(out_path))

    if len(pts) == 0:
        return ""
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, colors = pts[idx], colors[idx]

    hex_colors = [
        "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        for r, g, b in np.clip(colors, 0, 1)
    ]

    traces = [
        go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=hex_colors, opacity=0.7),
            name="3D Points",
        )
    ]
    if camera_centres is not None and len(camera_centres):
        traces.append(go.Scatter3d(
            x=camera_centres[:, 0], y=camera_centres[:, 1], z=camera_centres[:, 2],
            mode="markers+lines",
            marker=dict(size=8, color="red", symbol="diamond"),
            line=dict(color="red", width=2, dash="dash"),
            name="Cameras",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Hampi Revived — Interactive 3D Point Cloud (SfM)",
        scene=dict(
            bgcolor="#0d1117",
            xaxis=dict(backgroundcolor="#0d1117", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#0d1117", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#0d1117", gridcolor="#333"),
        ),
        paper_bgcolor="#0d1117",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")
    logger.info(f"Saved interactive plot: {out_path}")
    return out_path


# ─── 5. Top-down (orthographic) dense cloud slice ────────────────────────────

def plot_topdown(
    pts: np.ndarray,
    colors: np.ndarray,
    out_path: str = f"{OUT}/topdown_view.png",
    max_pts: int = 50000,
) -> str:
    _ensure(os.path.dirname(out_path))
    if len(pts) == 0:
        return ""
    if len(pts) > max_pts:
        idx = np.random.choice(len(pts), max_pts, replace=False)
        pts, colors = pts[idx], colors[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(pts[:, 0], pts[:, 2], c=np.clip(colors, 0, 1), s=0.5, alpha=0.5)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (depth, m)")
    ax.set_title("Top-Down Orthographic View — Hampi Dense Cloud", fontsize=13, fontweight="bold")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path


# ─── 6. Quality metrics dashboard ────────────────────────────────────────────

def plot_quality_dashboard(
    quality_data: List[dict],
    sfm_stats: dict,
    out_path: str = f"{OUT}/quality_dashboard.png",
) -> str:
    _ensure(os.path.dirname(out_path))
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Pipeline Quality Dashboard — Hampi Revived", fontsize=15, fontweight="bold", y=1.0)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    idxs = [q["idx"] for q in quality_data]
    blur = [q["blur_score"] for q in quality_data]
    bright = [q["brightness"] for q in quality_data]

    # Blur scores
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(idxs, blur, color=["#e74c3c" if b < 80 else "#2ecc71" for b in blur])
    ax1.axhline(80, color="orange", linestyle="--", linewidth=1.2, label="Threshold (80)")
    ax1.set_title("Sharpness (Laplacian var)", fontsize=10)
    ax1.set_xlabel("Image"); ax1.set_ylabel("Score")
    ax1.legend(fontsize=8)

    # Brightness
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(idxs, bright, color="#3498db")
    ax2.axhline(60, color="red", linestyle="--", linewidth=1)
    ax2.axhline(200, color="red", linestyle="--", linewidth=1)
    ax2.set_title("Mean Brightness", fontsize=10)
    ax2.set_xlabel("Image"); ax2.set_ylabel("Value (0-255)")

    # SfM stats
    ax3 = fig.add_subplot(gs[0, 2])
    sfm_labels = ["Images\nProcessed", "3D Points\n(×100)", "Cameras\nRegistered"]
    sfm_vals = [
        sfm_stats.get("n_images", 0),
        sfm_stats.get("n_points", 0) / 100,
        sfm_stats.get("n_cameras", 0),
    ]
    bars3 = ax3.bar(sfm_labels, sfm_vals, color=["#9b59b6", "#e67e22", "#1abc9c"])
    ax3.set_title("SfM Summary", fontsize=10)
    for bar, v, actual in zip(bars3, sfm_vals, [sfm_stats.get("n_images", 0), sfm_stats.get("n_points", 0), sfm_stats.get("n_cameras", 0)]):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 str(actual), ha="center", va="bottom", fontsize=9)

    # Keypoint counts
    ax4 = fig.add_subplot(gs[1, :2])
    kp_counts = [q.get("n_keypoints", 0) for q in quality_data]
    ax4.plot(idxs, kp_counts, "o-", color="#e67e22", linewidth=2, markersize=7)
    ax4.fill_between(idxs, kp_counts, alpha=0.25, color="#e67e22")
    ax4.set_title("SIFT Keypoints per Image", fontsize=10)
    ax4.set_xlabel("Image Index"); ax4.set_ylabel("# Keypoints")

    # Exposure pie
    ax5 = fig.add_subplot(gs[1, 2])
    exp_counts = {"good": 0, "underexposed": 0, "overexposed": 0}
    for q in quality_data:
        exp_counts[q.get("exposure", "good")] += 1
    ax5.pie(exp_counts.values(), labels=exp_counts.keys(),
            colors=["#2ecc71", "#e74c3c", "#f39c12"], autopct="%1.0f%%", startangle=90)
    ax5.set_title("Exposure Distribution", fontsize=10)

    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {out_path}")
    return out_path
