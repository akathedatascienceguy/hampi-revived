# Hampi Revived — Future Recommendations

**Project:** Hampi Revived
**Date:** March 2026
**Authors:** Yashvardhan Gupta · Nikita Gupta

---

## Current Limitations

The v1 inpainting pipeline produces plausible completions of damaged structures, but the generated tower does not perfectly match the archaeological specifics of the structure being restored:

- The stone texture, tier proportions, and carving density are generic Dravidian rather than site-specific Vijayanagara
- The damage boundary mask is a horizontal line, not a pixel-accurate outline of the damaged zones
- The model has no knowledge of what this specific monument looked like from historical records

The following recommendations address each of these gaps, ranked by expected impact.

---

## Recommendation 1 — LoRA Fine-Tuning on Hampi Architecture ⭐⭐⭐⭐⭐

**What:** Train a Low-Rank Adaptation (LoRA) layer on 50–200 photographs of Hampi monuments, then inject it into the SDXL inpainting model.

**Why it matters:** The base SDXL model knows "Indian temple" generically but has no specific knowledge of Vijayanagara architectural vocabulary: the specific sandstone oxidation colour, the double-tier cornice profiles used at Hampi, the density and style of mithunas and kirtimukhas in the carved frieze, the proportions of entrance gopurams in this specific complex.

A LoRA trained on even a small Hampi-specific dataset would anchor all generations to the right visual vocabulary — the inpainted top would have the correct stone, the correct tier spacing, the correct carving density for this specific site.

**How:**
1. Collect 50–200 images of Hampi monuments (Wikimedia Commons, ASI digital archive, field photography)
2. Caption each with monument name, period, and structural description
3. Fine-tune LoRA using `kohya_ss` or HuggingFace `diffusers` training scripts
4. Load LoRA weights at inference time: `pipeline.load_lora_weights("hampi_lora.safetensors")`

**Cost:** Free on Google Colab (T4 GPU, ~2–4 hours training)
**Expected output quality lift:** Very high — style fidelity would improve dramatically

---

## Recommendation 2 — Paired Damage→Complete Training Dataset ⭐⭐⭐⭐⭐

**What:** Build a dataset of (damaged image, intact image) pairs to teach the model the actual reconstruction task — not just architectural style.

**Why it matters:** The current model has never seen "here is a ruined gopuram → here is what it looked like intact." It generates from text descriptions of completeness. A model trained on paired examples learns the mapping from ruin patterns to completed structures directly, which is a fundamentally more powerful and accurate approach.

**Data sources:**
- Intact gopurams from the same Hampi complex photographed from matching angles
- ASI (Archaeological Survey of India) before/after restoration photographs
- Historical photographs from colonial-era surveys (c.1850–1920) showing earlier states of structures
- **Synthetic pairs:** Take intact monument photographs and programmatically damage them (randomly erase top sections, add degradation textures) to generate unlimited cheap supervision data

**Cost:** Primarily data collection effort. Training on free Colab GPU.
**Expected output quality lift:** Very high — enables the model to learn the actual task

---

## Recommendation 3 — Depth and Surface Normal Conditioning ⭐⭐⭐⭐

**What:** Replace or augment the Canny edge map with richer 3D-aware conditioning signals.

**Current problem:** Canny edges are flat 2D line maps — they lose all depth information. Two carved surfaces at very different depths (a column in the foreground vs. a tower tier in the background) look identical in a Canny map. This limits the structural accuracy of ControlNet-guided generations.

**Proposed approach:**
- **Depth maps** via MiDaS or DepthAnything-v2: encodes foreground/background depth of every carved element
- **Surface normals** via NormalBae: encodes 3D curvature of each carved surface, showing relief depth
- **MLSD line segments**: detects clean architectural geometry (horizontal tier lines, vertical columns) that Canny misses

These conditioning signals give the inpainting model a much richer prior on the actual 3D geometry of the surviving structure, so the completed portion aligns correctly in 3D space.

**Cost:** Free (all models available as HF Spaces or local inference)
**Effort:** Low — drop-in replacement for the Canny preprocessing step

---

## Recommendation 4 — Reference Image Conditioning via IP-Adapter ⭐⭐⭐⭐

**What:** Provide a reference photograph of a similar intact gopuram as a visual style anchor at inference time.

**Why it matters:** Text prompts are a lossy description of architectural style. "Sandstone, carved, Vijayanagara" in text cannot convey the exact hue of oxidised granite, the specific rhythm of repeated kirtimukha carvings, or the proportional relationship between the door jamb height and the tower. An image can.

IP-Adapter injects the reference image's CLIP image embedding directly into the cross-attention layers of the U-Net, alongside the text embedding:

```
Attention(Q, K_text+K_image, V_text+V_image)
```

The output image inherits the reference's visual style far more faithfully than any text description.

**Implementation:** `from diffusers import IPAdapterMixin` — free, works with existing SDXL pipeline.

**Cost:** Free
**Effort:** Low — add reference image loading + IP-Adapter weight injection to existing script

---

## Recommendation 5 — Pixel-Accurate Damage Segmentation ⭐⭐⭐

**What:** Replace the current horizontal-line damage boundary with a pixel-accurate mask produced by a fine-tuned segmentation model.

**Current problem:** The Laplacian variance sliding window detects a single horizontal cut across the full image width. In reality, structural damage is irregular — some areas of the top are still partially intact (remnant pilasters, partial cornice blocks) while some lower areas are degraded. A coarse horizontal mask over-inpaints intact regions and under-inpaints damaged ones.

**Proposed approach:**
- Fine-tune SAM (Segment Anything Model) on a small annotated dataset of Hampi damage patterns
- Or train a lightweight U-Net on manually annotated "damage masks" from 20–30 images
- The model outputs a per-pixel probability of structural damage → more precise inpainting mask → cleaner restoration

**Cost:** Free (SAM available via HF Spaces; annotation can be done with Label Studio)
**Effort:** Medium — requires data annotation

---

## Recommendation 6 — Multi-View Constrained Generation ⭐⭐⭐

**What:** Use photographs of the same structure from multiple angles to constrain what the completed version must look like in 3D — not just in a single 2D view.

**Why:** A gopuram restored from a single front-facing photograph might look correct from the front but be geometrically inconsistent from the side. Using the SfM point cloud already produced by the pipeline as a 3D constraint ensures the inpainted completion is geometrically consistent across views.

**Implementation:** NeRF-based inpainting (SPIn-NeRF or similar) — fill the missing volume in 3D space rather than in a 2D image. The result can be rendered from any angle.

---

## Implementation Roadmap

| Priority | Recommendation | Effort | Cost | Timeline |
|---|---|---|---|---|
| 1 | LoRA fine-tune on Hampi dataset | Medium | Free (Colab) | 2–3 weeks |
| 2 | Synthetic paired damage dataset | High (data collection) | Free | 4–6 weeks |
| 3 | Depth + normal conditioning | Low | Free | 1 week |
| 4 | IP-Adapter reference image | Low | Free | 3–5 days |
| 5 | SAM damage segmentation | Medium | Free | 2–3 weeks |
| 6 | Multi-view NeRF inpainting | High | Free/Low | 6–8 weeks |

The fastest high-impact path: **Depth conditioning (1 week) → IP-Adapter (1 week) → LoRA (3 weeks)** — all free, no paid compute required.

---

## Broader Vision

The combination of these improvements would move the project from "demonstrates the concept" to "archaeologically useful" — a restoration tool that could be used by ASI conservators and heritage researchers to visualise and document missing structural elements. The long-term ambition is a pipeline that ingests any photograph of a partially destroyed Hampi monument and produces a publication-quality architectural restoration, grounded in the specific Vijayanagara building canon and validated against iconographic records.

This would be among the first computational tools purpose-built for Deccan medieval architecture, filling a gap that currently no commercial or academic tool addresses.
