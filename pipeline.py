#!/usr/bin/env python3
"""
pipeline.py — Hampi Revived: End-to-End 3D Reconstruction Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stages:
  1. Data Ingestion        — download / generate Hampi images
  2. Preprocessing         — CLAHE, denoise, sharpen
  3. Feature Extraction    — SIFT detection + FLANN matching
  4. Structure from Motion — Essential matrix, pose recovery, triangulation
  5. Dense Reconstruction  — SGBM stereo → back-projected depth
  6. Mesh Reconstruction   — Poisson surface (via Open3D)
  7. Visualisation         — static PNGs + interactive HTML
  8. Groq Analysis         — AI-powered archaeological report

Run:
  python pipeline.py                     # full pipeline
  python pipeline.py --stage sfm         # single stage
  python pipeline.py --no-groq           # skip Groq
  python pipeline.py --images path/*.jpg # custom images
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml
import numpy as np

# ─── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("hampi")


# ─── Banner ───────────────────────────────────────────────────────────────────
BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║          H A M P I   R E V I V E D   🏛️                         ║
║   3D Reconstruction · Computer Vision · Data Science            ║
║   Stone meets Silicon · Vijayanagara Empire ~1336–1646 CE       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ─── Config ───────────────────────────────────────────────────────────────────
def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Stage printer ────────────────────────────────────────────────────────────
def stage(n: int, name: str, icon: str = "⬡"):
    logger.info(f"\n{'─'*60}")
    logger.info(f"  Stage {n}: {icon}  {name}")
    logger.info(f"{'─'*60}")


# ─── Pipeline ─────────────────────────────────────────────────────────────────
def run_pipeline(cfg: dict, args: argparse.Namespace) -> dict:
    results = {"stages": {}, "outputs": {}}
    t0 = time.time()

    # ── 1. Data Ingestion ──────────────────────────────────────────────────────
    stage(1, "Data Ingestion", "📷")
    from src.data_ingestion import download_images, load_images, dataset_stats

    if args.images:
        image_paths = list(args.images)
        logger.info(f"Using {len(image_paths)} user-supplied images.")
    else:
        image_paths = download_images(
            save_dir=cfg["data"]["raw_dir"],
            max_images=cfg["data"]["max_images"],
            target_size=tuple(cfg["data"]["target_size"]),
        )

    target_size = tuple(cfg["data"]["target_size"])
    images_raw = load_images(image_paths, target_size)
    stats = dataset_stats(images_raw)
    logger.info(f"Dataset: {stats}")
    results["stages"]["ingestion"] = stats

    # ── 2. Preprocessing ───────────────────────────────────────────────────────
    stage(2, "Preprocessing — CLAHE · Denoise · Sharpen", "🔬")
    from src.preprocessing import preprocess_batch

    images, quality = preprocess_batch(images_raw, denoise_imgs=True)
    logger.info(f"Preprocessed {len(images)} images.")
    results["stages"]["preprocessing"] = quality

    # ── 3. Feature Extraction ──────────────────────────────────────────────────
    stage(3, "Feature Extraction — SIFT + FLANN Matching", "🔑")
    from src.feature_extraction import (
        detect_and_describe, match_all_pairs,
        save_keypoints_plot, save_matches_plot,
        keypoint_stats, plot_keypoint_distribution,
    )

    detector = cfg["features"]["detector"]
    all_kps, all_descs = detect_and_describe(
        images,
        detector=detector,
        max_keypoints=cfg["features"]["max_keypoints"],
    )
    for i, q in enumerate(quality):
        q["n_keypoints"] = len(all_kps[i]) if i < len(all_kps) else 0

    matches_dict = match_all_pairs(
        all_descs,
        ratio=cfg["features"]["match_ratio_threshold"],
        min_matches=cfg["features"]["min_matches_for_pair"],
    )
    feat_stats = keypoint_stats(all_kps, matches_dict)
    logger.info(f"Feature stats: {feat_stats}")
    results["stages"]["features"] = feat_stats

    # Save keypoint / match visuals
    kp_paths = save_keypoints_plot(images, all_kps, cfg["outputs"]["features_dir"])
    match_paths = save_matches_plot(images, all_kps, matches_dict, cfg["outputs"]["features_dir"])
    dist_path = plot_keypoint_distribution(all_kps)
    results["outputs"]["keypoint_images"] = kp_paths
    results["outputs"]["match_images"] = match_paths
    results["outputs"]["keypoint_distribution"] = dist_path

    if not matches_dict:
        logger.error("No matching pairs found — cannot run SfM. Check image variety.")
        return results

    # ── 4. Structure from Motion ───────────────────────────────────────────────
    stage(4, "Structure from Motion — Triangulation & Camera Pose", "📐")
    from src.sfm import run_sfm, estimate_intrinsics

    pts3d_sparse, colors_sparse, camera_poses, cam_centres = run_sfm(
        images=images,
        all_kps=all_kps,
        matches_dict=matches_dict,
        img_shape=images[0].shape,
        focal_factor=cfg["sfm"]["focal_length_factor"],
        ransac_threshold=cfg["sfm"]["ransac_threshold"],
    )
    sfm_stats = {
        "n_images": len(images),
        "n_points": len(pts3d_sparse),
        "n_cameras": len(camera_poses),
        "n_pairs": len(matches_dict),
    }
    logger.info(f"SfM: {sfm_stats}")
    results["stages"]["sfm"] = sfm_stats

    # Save sparse cloud as numpy
    cloud_dir = cfg["outputs"]["point_clouds_dir"]
    Path(cloud_dir).mkdir(parents=True, exist_ok=True)
    np.save(f"{cloud_dir}/sparse_pts.npy", pts3d_sparse)
    np.save(f"{cloud_dir}/sparse_cols.npy", colors_sparse)
    results["outputs"]["sparse_cloud_npy"] = f"{cloud_dir}/sparse_pts.npy"

    # ── 5. Dense Reconstruction ────────────────────────────────────────────────
    stage(5, "Dense Reconstruction — SGBM Stereo Depth", "🌊")
    from src.dense_reconstruction import dense_reconstruct, voxel_downsample
    from src.sfm import estimate_intrinsics

    K = estimate_intrinsics(images[0].shape, cfg["sfm"]["focal_length_factor"])
    pts3d_dense, colors_dense = dense_reconstruct(
        images=images,
        camera_poses=camera_poses,
        K=K,
        max_pairs=min(8, len(camera_poses) - 1) if len(camera_poses) > 1 else 0,
    )

    if len(pts3d_dense) > 0:
        pts3d_dense, colors_dense = voxel_downsample(
            pts3d_dense, colors_dense, voxel_size=cfg["reconstruction"]["voxel_size"]
        )
        logger.info(f"Dense cloud: {len(pts3d_dense):,} points (after voxel downsample)")

    # Merge sparse + dense
    if len(pts3d_dense) > 0 and len(pts3d_sparse) > 0:
        pts_all = np.vstack([pts3d_sparse, pts3d_dense])
        cols_all = np.vstack([colors_sparse, colors_dense])
    elif len(pts3d_sparse) > 0:
        pts_all, cols_all = pts3d_sparse, colors_sparse
    else:
        pts_all, cols_all = pts3d_dense, colors_dense

    results["stages"]["dense"] = {
        "n_dense_pts": len(pts3d_dense),
        "n_total_pts": len(pts_all),
    }
    logger.info(f"Combined cloud: {len(pts_all):,} points")

    # ── 6. Mesh Reconstruction ─────────────────────────────────────────────────
    stage(6, "Mesh Reconstruction — Poisson Surface", "🕸️")
    from src.mesh import run_mesh_pipeline

    mesh_result = run_mesh_pipeline(
        pts=pts_all,
        colors=cols_all,
        out_dir_pcd=cfg["outputs"]["point_clouds_dir"],
        out_dir_mesh=cfg["outputs"]["meshes_dir"],
        voxel_size=cfg["reconstruction"]["voxel_size"],
        poisson_depth=cfg["reconstruction"]["poisson_depth"],
    )
    results["stages"]["mesh"] = mesh_result
    results["outputs"].update(mesh_result)

    # ── 7. Visualisation ───────────────────────────────────────────────────────
    stage(7, "Visualisation — Static + Interactive", "🎨")
    from src.visualization import (
        plot_image_grid,
        plot_match_matrix,
        plot_sparse_cloud_3d,
        plot_sparse_cloud_plotly,
        plot_topdown,
        plot_quality_dashboard,
    )

    vis_dir = cfg["outputs"]["visualizations_dir"]
    Path(vis_dir).mkdir(parents=True, exist_ok=True)

    grid_path = plot_image_grid(
        images_raw[:8],
        title="Hampi — Input Images",
        out_path=f"{vis_dir}/image_grid.png",
    )
    matrix_path = plot_match_matrix(
        len(images), matches_dict, out_path=f"{vis_dir}/match_matrix.png"
    )
    cloud_3d_path = plot_sparse_cloud_3d(
        pts_all, cols_all, cam_centres,
        out_path=f"{vis_dir}/sparse_cloud_3d.png",
    )
    interactive_path = plot_sparse_cloud_plotly(
        pts_all, cols_all, cam_centres,
        out_path=f"{vis_dir}/interactive_cloud.html",
    )
    topdown_path = plot_topdown(
        pts_all, cols_all,
        out_path=f"{vis_dir}/topdown_view.png",
    )
    dash_path = plot_quality_dashboard(
        quality, sfm_stats, out_path=f"{vis_dir}/quality_dashboard.png"
    )
    results["outputs"]["visualizations"] = {
        "image_grid": grid_path,
        "match_matrix": matrix_path,
        "3d_cloud": cloud_3d_path,
        "interactive_html": interactive_path,
        "topdown": topdown_path,
        "quality_dashboard": dash_path,
    }

    # ── 8. Groq Analysis ───────────────────────────────────────────────────────
    if not args.no_groq:
        stage(8, "Groq AI — Archaeological Analysis", "🤖")
        from src.groq_analysis import GroqArchaeologist, save_analyses

        agent = GroqArchaeologist()
        image_analyses = agent.analyse_batch(images_raw, n_images=min(4, len(images_raw)))
        site_report = agent.generate_site_report(image_analyses, sfm_stats)

        paths = save_analyses(image_analyses, site_report, cfg["outputs"]["reports_dir"])
        results["stages"]["groq"] = {"n_analysed": len(image_analyses)}
        results["outputs"]["reports"] = paths

        logger.info(f"\n{'='*60}")
        logger.info("SITE REPORT (excerpt):")
        logger.info("=" * 60)
        logger.info(site_report[:1200] + ("…" if len(site_report) > 1200 else ""))
    else:
        logger.info("Groq analysis skipped (--no-groq flag).")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info(f"\n{'═'*60}")
    logger.info(f"  PIPELINE COMPLETE  in {elapsed:.1f}s")
    logger.info(f"{'═'*60}")
    logger.info(f"  Images:      {len(images)}")
    logger.info(f"  3D Points:   {len(pts_all):,}")
    logger.info(f"  Cameras:     {len(camera_poses)}")
    if mesh_result:
        n_tri = mesh_result.get('n_triangles')
        logger.info(f"  Triangles:   {n_tri:,}" if n_tri else f"  Mesh:        {list(mesh_result.keys())}")
    logger.info(f"\n  Key outputs:")
    logger.info(f"  • Visualizations : {vis_dir}/")
    logger.info(f"  • Point clouds   : {cfg['outputs']['point_clouds_dir']}/")
    logger.info(f"  • Meshes         : {cfg['outputs']['meshes_dir']}/")
    logger.info(f"  • Reports        : {cfg['outputs']['reports_dir']}/")
    logger.info(f"  • Interactive 3D : {interactive_path}")

    # Save full results JSON
    report_dir = cfg["outputs"]["reports_dir"]
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    json_path = f"{report_dir}/pipeline_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  • Results JSON   : {json_path}")

    return results


# ─── CLI ──────────────────────────────────────────────────────────────────────
def main():
    print(BANNER)

    parser = argparse.ArgumentParser(description="Hampi Revived — 3D Reconstruction Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--images", nargs="+", help="Custom image paths (skip download)")
    parser.add_argument("--no-groq", action="store_true", help="Skip Groq AI analysis")
    parser.add_argument("--stage", choices=["ingest", "preprocess", "features", "sfm", "dense", "mesh", "vis", "groq"],
                        help="Run only up to this stage (for debugging)")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)

    # Load .env if present
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    run_pipeline(cfg, args)


if __name__ == "__main__":
    main()
