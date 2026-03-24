# Hampi Revived — Data Science Technical Report

**Project:** Hampi Revived
**Date:** March 2026
**Authors:** Yashvardhan Gupta · Nikita Gupta

---

## 1. Image Preprocessing — Classical Computer Vision

### 1.1 CLAHE (Contrast Limited Adaptive Histogram Equalization)

Operates in LAB colour space. The L (luminance) channel is split into an 8×8 tile grid. Within each tile, a local histogram is computed and clipped at `clipLimit=2.0` — pixels above the clip threshold are redistributed uniformly across all intensity bins, preventing noise amplification while boosting local contrast. The result is reassembled and converted back to BGR.

**Why LAB, not RGB:** Applying histogram equalisation directly to RGB channels distorts colour. LAB separates luminance (L) from chrominance (A, B), so stone colour is preserved while carved-edge contrast increases.

### 1.2 Non-Local Means Denoising

`fastNlMeansDenoisingColored` with `h=6.0`, template window 7×7, search window 21×21.

For each pixel p, a 7×7 patch is compared against all 7×7 patches within a 21×21 neighbourhood. The denoised value is a weighted average:

```
NLM(p) = Σ_q  w(p,q) · I(q)
w(p,q) = exp(−||P(p) − P(q)||² / h²)
```

High-similarity patches get exponentially higher weight. This preserves structural edges (every patch near an edge is dissimilar to patches in flat regions, so it gets low weight and is not averaged away) while averaging out random sensor noise.

### 1.3 Unsharp Masking

```
output = 1.6 × original − 0.6 × GaussianBlur(σ=3)
```

The Gaussian blur removes high-frequency content. Subtracting it from the original isolates only the high frequencies (edges, texture). Adding these back amplified (×1.6 forward, −0.6 blur) sharpens carved-stone relief detail.

**Measured effect:** Laplacian variance (sharpness proxy) increased by **177%** on the gopuram image (1,795 → 4,978).

---

## 2. Feature Extraction — SIFT + FLANN

### 2.1 SIFT (Scale-Invariant Feature Transform)

For each image, SIFT detects up to 5,000 keypoints. The algorithm:

1. Builds a Gaussian scale-space pyramid (different σ levels per octave)
2. Computes Difference-of-Gaussian (DoG) between adjacent scales
3. Finds local extrema in DoG → candidate keypoints, stable across scale
4. Removes low-contrast candidates (threshold 0.04) and edge responses (Harris ratio)
5. Assigns dominant gradient orientation from a 36-bin histogram in a 16×16 neighbourhood
6. Computes a 128-dim descriptor: 4×4 spatial grid of 8-bin gradient histograms

SIFT is rotation-, scale-, and partial-illumination-invariant — critical for Hampi's carved-granite surfaces photographed under varying sunlight and angles.

**Pipeline results:**

| Metric | Value |
|---|---|
| Mean keypoints/image | 4,123 |
| Min keypoints (blurry image) | 406 |
| Max keypoints | 5,001 |

### 2.2 FLANN Matching + Lowe's Ratio Test

FLANN (Fast Library for Approximate Nearest Neighbours) with a KD-tree index (5 trees, 50 checks) finds the 2 nearest descriptor matches for each keypoint. Lowe's ratio test keeps a match only if:

```
distance(best_match) < 0.75 × distance(second_best_match)
```

This rejects ambiguous matches (where two scene points look similar). A match is accepted only when one candidate is clearly better than all others.

**Pipeline results:**

| Metric | Value |
|---|---|
| Connected image pairs (≥20 matches) | 47 |
| Mean matches per pair | 32.7 |
| Max matches per pair | 80 |

---

## 3. Structure from Motion (SfM)

### 3.1 Essential Matrix Estimation

Given matched point pairs (p₁, p₂) between two images with calibration matrix K (estimated from EXIF focal length or default approximation):

```
x₂ᵀ E x₁ = 0
E = [t]× R
```

E is the 3×3 Essential Matrix encoding both the rotation R and translation t between cameras. Estimated using the 5-point algorithm inside RANSAC (1000 iterations, reprojection threshold 1.0px) to reject outlier matches.

### 3.2 Camera Pose Recovery & Triangulation

R and t are recovered from E via SVD decomposition (4 candidate solutions, the correct one has all triangulated points in front of both cameras). Triangulation uses the DLT (Direct Linear Transform) method:

```
λ₁ x₁ = P₁ X,  λ₂ x₂ = P₂ X
→ AX = 0, solved via SVD
```

**Sparse reconstruction results:**

| Metric | Value |
|---|---|
| 3D points triangulated | 69 |
| Camera poses recovered | 6 of 12 |
| Image pairs with valid geometry | 47 |

---

## 4. Dense Reconstruction — SGBM Stereo Depth

SGBM (Semi-Global Block Matching) computes a dense disparity map between stereo image pairs by minimising an energy function:

```
E(D) = Σ_p C(p, D_p) + Σ_{q∈N_p} P₁·[|D_p − D_q| = 1] + P₂·[|D_p − D_q| > 1]
```

- C(p, D_p): data cost (pixel dissimilarity at disparity D)
- P₁ = 8: small disparity changes penalised lightly (smooth surfaces)
- P₂ = 32: large disparity jumps penalised heavily (depth discontinuities preserved)

Disparity is back-projected to 3D: `Z = f·B/d` where f=focal length, B=baseline, d=disparity.

**Dense reconstruction results:**

| Metric | Value |
|---|---|
| Dense points generated | 106,525 |
| Total point cloud (sparse + dense) | 106,594 |

---

## 5. Poisson Surface Reconstruction

Open3D's Poisson surface reconstruction fits a watertight triangle mesh to the oriented point cloud by solving a Poisson equation:

```
∇²χ = ∇·V
```

where V is the vector field defined by point normals, χ is an indicator function, and the mesh is extracted at an isosurface. Octree depth=9.

**Mesh output:** `outputs/meshes/hampi_cloud_trimesh.ply` — 106,594 points, watertight surface.

---

## 6. Generative AI — SDXL Inpainting

### 6.1 Latent Diffusion Model (LDM) Foundation

SDXL encodes images into a compressed latent space via a VAE:

```
z = E(x),  z ∈ ℝ^(H/8 × W/8 × 4)
```

An 8× spatial compression reduces a 768×512 image to a 96×64×4 latent. A U-Net denoiser ε_θ(z_t, t, c) is trained to predict the noise added at timestep t, conditioned on text embedding c. At inference, starting from Gaussian noise z_T, the model iteratively denoises:

```
z_{t-1} = f(z_t, ε_θ(z_t, t, c))
```

guided by classifier-free guidance at scale 8.0:

```
ε_guided = ε_uncond + 8.0 × (ε_cond − ε_uncond)
```

### 6.2 Inpainting Mechanism

The inpainting variant conditions the U-Net on both the masked image and the binary mask channel concatenated to the noisy latent. At each denoising step, unmasked pixels are replaced with their noised originals — the model cannot modify the intact lower structure. Only the masked region (damaged top) is free to evolve.

This is why the doorway, flanking carved walls, and stone color in the output are pixel-identical to the input: they are mathematically clamped at every denoising step.

### 6.3 Damage Boundary Detection

A Laplacian variance sliding window (30-row bands) scans the image top-to-bottom:

```
var_r = Var(|∇²I|[r:r+30, :])
```

The row with the steepest drop in variance (largest negative first-difference in the upper half) marks the transition from "rich carved detail" to "rubble/missing structure." Detected boundary: **row 330/800 (41% from top)** on the gopuram image.

### 6.4 SSIM-Based Candidate Ranking

When multiple generations are produced, the best is selected by Structural Similarity Index (SSIM) between the generated image's Canny map and the original image's Canny map:

```
SSIM(x, y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
```

where l=luminance, c=contrast, s=structure similarity, each measured in 11×11 Gaussian windows. Higher SSIM = generated structure more closely mirrors the original layout.

**Best candidate SSIM (SD3, cached):** 0.4006

### 6.5 Cosine-Feathered Compositing

To blend the inpainted top with the original base:

```python
alpha(r) = 0.5 × (1 + cos(π × t))   # t ∈ [0,1] across blend band
output(r) = inpainted(r) × alpha(r) + original(r) × (1 − alpha(r))
```

The cosine schedule gives a perceptually smooth S-curve transition (no linear seam artefacts) over an 80-pixel blend band centred on the damage boundary.

---

## 7. Quantitative Results Summary

| Stage | Key Metric | Value |
|---|---|---|
| Preprocessing | Sharpness improvement (Laplacian var) | +177% |
| Preprocessing | Exposure quality | Good on 11/12 images |
| SIFT | Mean keypoints per image | 4,123 |
| Matching | Connected image pairs | 47 / 66 possible |
| SfM | 3D points triangulated | 69 sparse |
| Dense | Dense point cloud | 106,525 pts |
| Inpainting SSIM | Structure match vs original | 0.4006 |
| Damage boundary | Detection accuracy | Row 330/800 (visual inspection ✓) |
| Compute cost | All stages combined | $0 (free APIs + local) |
