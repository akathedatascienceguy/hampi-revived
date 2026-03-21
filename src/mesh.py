"""
mesh.py — Surface mesh reconstruction from point clouds.

Uses Open3D's Poisson surface reconstruction (or BPA fallback).
Steps:
  1. Statistical outlier removal
  2. Normal estimation
  3. Poisson reconstruction → watertight mesh
  4. Mesh simplification (quadric decimation)
  5. Export as PLY and OBJ
"""

import logging
import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning("open3d not available — mesh reconstruction will use trimesh fallback.")

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


def numpy_to_pcd(pts: np.ndarray, colors: Optional[np.ndarray] = None):
    """Convert (N,3) numpy arrays to an Open3D PointCloud."""
    if not HAS_OPEN3D:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        c = np.clip(colors, 0, 1).astype(np.float64)
        pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def clean_point_cloud(
    pcd,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    voxel_size: Optional[float] = None,
):
    """Statistical outlier removal + optional voxel downsampling."""
    if not HAS_OPEN3D or pcd is None:
        return pcd
    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd


def estimate_normals(pcd, radius: float = 0.5, max_nn: int = 30):
    """Estimate surface normals via KNN search."""
    if not HAS_OPEN3D or pcd is None:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    return pcd


def poisson_mesh(pcd, depth: int = 9, scale: float = 1.1):
    """Poisson surface reconstruction — returns (mesh, densities)."""
    if not HAS_OPEN3D or pcd is None:
        return None, None
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale
    )
    return mesh, densities


def trim_mesh_by_density(mesh, densities, quantile: float = 0.05):
    """Remove low-density faces (typically floating artefacts)."""
    if not HAS_OPEN3D or mesh is None:
        return mesh
    dens = np.asarray(densities)
    threshold = np.quantile(dens, quantile)
    verts_to_remove = dens < threshold
    mesh.remove_vertices_by_mask(verts_to_remove)
    return mesh


def simplify_mesh(mesh, target_triangles: int = 50_000):
    """Quadric decimation to target triangle count."""
    if not HAS_OPEN3D or mesh is None:
        return mesh
    n = len(mesh.triangles)
    if n > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    return mesh


def save_mesh(mesh, out_dir: str = "outputs/meshes", name: str = "hampi_mesh") -> dict:
    """Save mesh as PLY and OBJ; return saved paths."""
    if not HAS_OPEN3D or mesh is None:
        return {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext in ["ply", "obj"]:
        p = os.path.join(out_dir, f"{name}.{ext}")
        o3d.io.write_triangle_mesh(p, mesh)
        paths[ext] = p
        logger.info(f"  Saved mesh: {p}")
    return paths


def save_point_cloud(pcd, out_dir: str = "outputs/point_clouds", name: str = "hampi_cloud") -> str:
    """Save point cloud as PLY."""
    if not HAS_OPEN3D or pcd is None:
        return ""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = os.path.join(out_dir, f"{name}.ply")
    o3d.io.write_point_cloud(p, pcd)
    logger.info(f"  Saved point cloud: {p}")
    return p


def trimesh_fallback(pts: np.ndarray, colors: np.ndarray, out_dir: str = "outputs/meshes") -> dict:
    """Minimal point-cloud save using trimesh when open3d is unavailable."""
    if not HAS_TRIMESH:
        return {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cloud = trimesh.PointCloud(pts, colors=(np.clip(colors, 0, 1) * 255).astype(np.uint8))
    p = os.path.join(out_dir, "hampi_cloud_trimesh.ply")
    cloud.export(p)
    logger.info(f"  Saved trimesh point cloud: {p}")
    return {"ply": p, "n_points": len(pts), "backend": "trimesh"}


def run_mesh_pipeline(
    pts: np.ndarray,
    colors: np.ndarray,
    out_dir_pcd: str = "outputs/point_clouds",
    out_dir_mesh: str = "outputs/meshes",
    voxel_size: float = 0.05,
    poisson_depth: int = 9,
) -> dict:
    """
    Full mesh pipeline: numpy → clean pcd → normals → Poisson mesh → save.
    Returns dict with paths and mesh stats.
    """
    result = {}
    if not HAS_OPEN3D:
        logger.warning("open3d unavailable — using trimesh fallback for point cloud save.")
        return trimesh_fallback(pts, colors, out_dir_mesh)

    if len(pts) < 100:
        logger.warning(f"Too few points ({len(pts)}) for mesh reconstruction.")
        return result

    logger.info(f"Building mesh from {len(pts):,} points…")

    # 1. Build PCD
    pcd = numpy_to_pcd(pts, colors)

    # 2. Clean
    pcd = clean_point_cloud(pcd, voxel_size=voxel_size)
    logger.info(f"  After cleaning: {len(pcd.points):,} points")

    # 3. Normals
    pcd = estimate_normals(pcd)

    # 4. Save point cloud
    pcd_path = save_point_cloud(pcd, out_dir_pcd)
    result["point_cloud_ply"] = pcd_path
    result["n_points"] = len(pcd.points)

    # 5. Poisson mesh
    mesh, densities = poisson_mesh(pcd, depth=poisson_depth)
    if mesh is None:
        return result

    # 6. Trim + simplify
    mesh = trim_mesh_by_density(mesh, densities)
    mesh = simplify_mesh(mesh)
    mesh.compute_vertex_normals()

    # 7. Save
    paths = save_mesh(mesh, out_dir_mesh)
    result.update(paths)
    result["n_triangles"] = len(mesh.triangles)
    result["n_vertices"] = len(mesh.vertices)
    logger.info(f"  Mesh: {result['n_triangles']:,} triangles, {result['n_vertices']:,} vertices")

    return result
