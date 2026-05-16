"""
collect_gopuram_images.py — Strict Hampi-only image collection.

Sources:
  1. Wikimedia Commons categories scoped to Hampi monuments
  2. Historical/colonial-era archive searches (Survey of India, ASI)
  3. Specific broken-monument searches all anchored to "Hampi"

Auto-rejects:
  - Images where mean HSV saturation > 60  (colourful Tamil stucco temples)
  - Extreme aspect ratios  (panorama strips, portrait banners)
  - Images < 300px on shortest side or < 80 KB

Run:
    source venv/bin/activate
    python collect_gopuram_images.py
"""

import hashlib, time, io
import requests
import numpy as np
from pathlib import Path
from PIL import Image

ROOT     = Path(__file__).parent
RAW_DIR  = ROOT / "data" / "raw"
REF_DIR  = ROOT / "data" / "reference"
API      = "https://commons.wikimedia.org/w/api.php"
HEADERS  = {"User-Agent": "HampiRevived/2.0 (research; yvg1799@gmail.com)"}

MIN_BYTES     = 80_000
MAX_DIM       = 2000
MAX_SAT       = 60     # reject colourful stucco (Meenakshi etc.)
MIN_AR, MAX_AR = 0.30, 3.0   # reject extreme panoramas / banners

RAW_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)


# ─── Quality gate ──────────────────────────────────────────────────────────────

def is_hampi_style(data: bytes) -> bool:
    """
    Reject if mean HSV saturation is too high (colourful Tamil stucco)
    or aspect ratio is out of range.
    """
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        ar = w / h
        if ar > MAX_AR or ar < MIN_AR:
            return False
        # Saturation check
        arr = np.array(img.resize((64, 64))).astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        sat = np.where(cmax > 0, (cmax - cmin) / cmax, 0)
        mean_sat = sat.mean() * 100
        if mean_sat > MAX_SAT:
            return False
        return True
    except Exception:
        return False


# ─── Wikimedia helpers ─────────────────────────────────────────────────────────

def category_files(category: str, limit: int = 30) -> list[str]:
    cat = "Category:" + category.replace(" ", "_")
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": cat, "cmtype": "file",
        "cmlimit": limit * 3, "format": "json",
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        titles = [m["title"] for m in r.json().get("query", {}).get("categorymembers", [])
                  if m.get("title", "").lower().endswith((".jpg", ".jpeg", ".png"))]
        return titles[:limit]
    except Exception as e:
        print(f"    category error: {e}")
        return []


def search_files(query: str, limit: int = 20) -> list[str]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(limit * 4, 50),
        "prop": "imageinfo",
        "iiprop": "url|size",
        "iiurlwidth": MAX_DIM,
        "format": "json",
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=25)
        r.raise_for_status()
        titles = [
            p.get("title", "") for p in r.json().get("query", {}).get("pages", {}).values()
            if p.get("title", "").lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        return titles[:limit]
    except Exception as e:
        print(f"    search error: {e}")
        return []


def image_url(title: str) -> str | None:
    params = {
        "action": "query", "titles": title,
        "prop": "imageinfo", "iiprop": "url|size",
        "iiurlwidth": MAX_DIM, "format": "json",
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        for page in r.json().get("query", {}).get("pages", {}).values():
            ii = page.get("imageinfo", [{}])[0]
            w, h = ii.get("width", 0), ii.get("height", 0)
            if w > 300 and h > 300:
                return ii.get("thumburl") or ii.get("url")
    except Exception as e:
        print(f"    URL error: {e}")
    return None


def stem_for(title: str) -> str:
    name = title.split("File:")[-1]
    return hashlib.md5(name.encode()).hexdigest()[:10]


def download_image(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        print(f"    cached  {dest.name}")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=40, stream=True)
        r.raise_for_status()
        data = b"".join(r.iter_content(8192))
        if len(data) < MIN_BYTES:
            return False
        if not is_hampi_style(data):
            return False
        img = Image.open(io.BytesIO(data)).convert("RGB")
        if min(img.size) < 300:
            return False
        dest.write_bytes(data)
        kb = len(data) // 1024
        print(f"    ✓  {dest.name}  ({kb} KB  {img.size[0]}×{img.size[1]})")
        return True
    except Exception as e:
        print(f"    ✗  {e}")
        return False


def fetch(titles: list[str], dest_dir: Path, limit: int, tag: str) -> int:
    print(f"\n  [{tag}]")
    count = 0
    for title in titles:
        if count >= limit:
            break
        url = image_url(title)
        if not url:
            continue
        ext = ".jpg" if any(x in url.lower() for x in (".jpg", ".jpeg")) else ".png"
        dest = dest_dir / (stem_for(title) + ext)
        if download_image(url, dest):
            count += 1
        time.sleep(0.3)
    print(f"  → {count} saved")
    return count


# ─── Strict Hampi-only sources ─────────────────────────────────────────────────
#
# All searches include the word "Hampi" or use a known Hampi Commons category.
# No generic South Indian temple searches — those contaminate with Tamil Nadu.

CATEGORY_FETCHES = [
    # Wikimedia Commons categories scoped exactly to Hampi structures
    ("Virupaksha Temple, Hampi",          RAW_DIR, 10, "Cat — Virupaksha Temple"),
    ("Vittala Temple",                    RAW_DIR,  8, "Cat — Vittala Temple"),
    ("Hazara Rama Temple",                RAW_DIR,  8, "Cat — Hazara Rama Temple"),
    ("Hemakuta Hill",                     RAW_DIR,  6, "Cat — Hemakuta Hill"),
    ("Mahanavami Dibba",                  RAW_DIR,  4, "Cat — Mahanavami Dibba"),
    ("Lotus Mahal",                       RAW_DIR,  4, "Cat — Lotus Mahal"),
    ("Elephant Stables, Hampi",           RAW_DIR,  4, "Cat — Elephant Stables"),
    ("Achyutaraya Temple",                RAW_DIR,  5, "Cat — Achyutaraya Temple"),
    ("Krishnadeva Raya",                  RAW_DIR,  3, "Cat — Krishna Temple"),
]

KEYWORD_SEARCHES = [
    # Gopuram-specific, always anchored to Hampi
    ("Hampi gopuram ruins Karnataka",           RAW_DIR,  6, "Hampi — gopuram ruins"),
    ("Virupaksha Hampi gopuram tower",          RAW_DIR,  6, "Hampi — Virupaksha gopuram"),
    ("Hazara Rama Hampi gopuram damaged",       RAW_DIR,  5, "Hampi — Hazara Rama gopuram"),
    ("Hampi north gopuram entrance broken",     RAW_DIR,  5, "Hampi — north gopuram"),
    ("Vittala temple Hampi ruins columns",      RAW_DIR,  5, "Hampi — Vittala ruins"),
    ("Achyutaraya temple Hampi ruined",         RAW_DIR,  4, "Hampi — Achyutaraya"),
    # Historical / archival
    ("Hampi ruins photograph 1856 Fergusson",   RAW_DIR,  4, "Archive — Fergusson 1856"),
    ("Hampi Vijayanagara ruins 1900 photograph",RAW_DIR,  5, "Archive — colonial photos"),
    ("Hampi ruins Survey of India photograph",  RAW_DIR,  4, "Archive — Survey of India"),
    ("Hampi ruins ASI archaeological photograph",RAW_DIR, 4, "Archive — ASI photos"),
    ("Hampi Vijayanagara historical engraving", RAW_DIR,  3, "Archive — historical engravings"),
    # Broken significant monuments — all anchored to Hampi
    ("Hampi broken pillar mandapa ruins",       RAW_DIR,  4, "Hampi — ruined mandapas"),
    ("Hampi stone chariot Vittala ruined",      RAW_DIR,  4, "Hampi — stone chariot"),
    ("Hampi Tungabhadra ruins boulders",        RAW_DIR,  4, "Hampi — riverside ruins"),
    ("Hampi Vijayanagara carved frieze wall",   RAW_DIR,  4, "Hampi — carved walls"),
]

# Reference: only Virupaksha intact gopuram (on the same Hampi site)
REFERENCE_SEARCHES = [
    ("Virupaksha Hampi gopuram intact tower",   REF_DIR,  6, "Ref — Virupaksha intact"),
    ("Hampi Virupaksha tower complete",         REF_DIR,  4, "Ref — Virupaksha complete"),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("Hampi Revived — Strict Hampi-Only Image Collection")
    print("  Saturation filter: rejects colourful Tamil stucco temples")
    print("  Aspect filter: rejects panorama strips & banners")
    print("=" * 62)

    total = 0

    print("\n── CATEGORY-BASED (exact Hampi Commons categories) ──")
    for cat, dest, lim, tag in CATEGORY_FETCHES:
        titles = category_files(cat, limit=lim * 3)
        total += fetch(titles, dest, lim, tag)

    print("\n── KEYWORD SEARCHES (all anchored to 'Hampi') ──")
    for query, dest, lim, tag in KEYWORD_SEARCHES:
        titles = search_files(query, limit=lim * 3)
        total += fetch(titles, dest, lim, tag)

    print("\n── REFERENCE IMAGES (Virupaksha intact, same site) ──")
    for query, dest, lim, tag in REFERENCE_SEARCHES:
        titles = search_files(query, limit=lim * 3)
        total += fetch(titles, dest, lim, tag)

    n_raw = len(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))
    n_ref = len(list(REF_DIR.glob("*.jpg")) + list(REF_DIR.glob("*.png")))
    print(f"\n{'='*62}")
    print(f"Collection complete — {total} new downloads")
    print(f"  data/raw/       : {n_raw} images")
    print(f"  data/reference/ : {n_ref} images")
    print(f"\nNext: python lora_pipeline.py")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
