"""
groq_analysis.py — AI-powered archaeological analysis using Groq.

Uses:
  • llama-3.2-11b-vision-preview  → visual analysis of monument images
  • llama-3.3-70b-versatile       → text reasoning, historical context, report gen

Capabilities:
  1. Identify monument type and architectural style
  2. Detect structural damage / weathering
  3. Estimate architectural period (Vijayanagara Empire ~1336–1646 CE)
  4. Generate a structured site report
  5. Answer free-form archaeological queries
"""

import base64
import json
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("groq package not installed.")

SYSTEM_PROMPT = """You are an expert archaeologist specialising in the Vijayanagara Empire
(1336–1646 CE) and the ruins of Hampi, Karnataka, India. You combine rigorous archaeological
analysis with deep knowledge of Dravidian and Chalukya architectural traditions.

When given an image of Hampi ruins, you:
1. Identify the monument/structure type (temple, gopura, mandapa, chariot, elephant stable, etc.)
2. Describe visible architectural features (columns, friezes, carvings, inscriptions)
3. Assess structural condition and weathering (granite, schist, laterite)
4. Estimate construction period within Vijayanagara chronology
5. Note culturally significant iconography (Vishnu, Shiva, Narasimha, etc.)

Be precise, scholarly, and cite parallels to known Hampi monuments where appropriate."""


def _encode_image(img: np.ndarray, quality: int = 70) -> str:
    """Encode a numpy BGR image as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _encode_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


class GroqArchaeologist:
    """Groq-powered archaeological analysis agent."""

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key or not HAS_GROQ:
            self.client = None
            if not HAS_GROQ:
                logger.warning("groq not installed — analysis will be skipped.")
            else:
                logger.warning("GROQ_API_KEY not set — analysis will be skipped.")
        else:
            self.client = Groq(api_key=key)

    def analyse_image(
        self,
        img: np.ndarray,
        prompt: str = "Analyse this Hampi ruins image. Identify the structure, its architectural features, condition, and cultural significance.",
        model: str = "llama-3.2-11b-vision-preview",
        max_tokens: int = 800,
    ) -> str:
        """Send an image to Groq vision model and return analysis."""
        if self.client is None:
            return "[Groq unavailable — set GROQ_API_KEY to enable analysis]"

        b64 = _encode_image(img)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq vision error: {e}")
            return f"[Analysis failed: {e}]"

    def analyse_batch(
        self,
        images: list,
        n_images: int = 4,
    ) -> list[dict]:
        """Analyse first n images; return list of {idx, analysis}."""
        results = []
        for i, img in enumerate(images[:n_images]):
            logger.info(f"  Groq: analysing image {i}…")
            text = self.analyse_image(img)
            results.append({"idx": i, "analysis": text})
        return results

    def generate_site_report(
        self,
        image_analyses: list[dict],
        sfm_stats: dict,
        model: str = "llama-3.3-70b-versatile",
    ) -> str:
        """
        Synthesise individual image analyses + 3D reconstruction stats
        into a structured archaeological site report.
        """
        if self.client is None:
            return "[Groq unavailable]"

        analyses_text = "\n\n".join(
            f"Image {r['idx']}:\n{r['analysis']}" for r in image_analyses
        )
        prompt = f"""
You have received the following image-level archaeological analyses of the
Hampi ruins site, along with 3D reconstruction metrics.

--- IMAGE ANALYSES ---
{analyses_text}

--- 3D RECONSTRUCTION METRICS ---
{json.dumps(sfm_stats, indent=2)}

Generate a structured **Archaeological Site Report** with the following sections:
1. Executive Summary
2. Identified Structures & Monuments
3. Architectural Period & Style
4. Structural Condition Assessment
5. Notable Iconography & Inscriptions
6. 3D Reconstruction Quality Assessment
7. Recommendations for Field Survey / Conservation

Use formal archaeological language. Reference the Vijayanagara Empire, Krishnadevaraya
(1509–1529 CE), and relevant architectural parallels where appropriate.
"""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq report error: {e}")
            return f"[Report generation failed: {e}]"

    def ask(self, question: str, context: str = "", model: str = "llama-3.3-70b-versatile") -> str:
        """Free-form Q&A about Hampi / Vijayanagara archaeology."""
        if self.client is None:
            return "[Groq unavailable]"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if context:
            messages.append({"role": "user", "content": f"Context:\n{context}"})
            messages.append({"role": "assistant", "content": "Understood. Please ask your question."})
        messages.append({"role": "user", "content": question})
        try:
            r = self.client.chat.completions.create(model=model, messages=messages, max_tokens=800)
            return r.choices[0].message.content
        except Exception as e:
            return f"[Error: {e}]"


def save_analyses(analyses: list[dict], report: str, out_dir: str = "outputs/reports") -> dict:
    """Save individual analyses and the site report to text files."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = {}

    for a in analyses:
        p = os.path.join(out_dir, f"analysis_image_{a['idx']:02d}.txt")
        with open(p, "w") as f:
            f.write(a["analysis"])
        paths[f"image_{a['idx']}"] = p

    report_path = os.path.join(out_dir, "site_report.md")
    with open(report_path, "w") as f:
        f.write("# Hampi Revived — Archaeological Site Report\n\n")
        f.write(report)
    paths["site_report"] = report_path
    logger.info(f"  Saved site report: {report_path}")
    return paths
