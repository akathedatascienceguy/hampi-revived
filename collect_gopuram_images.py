"""
collect_gopuram_images.py — Download intact South Indian gopuram images.

Downloads two batches via the Wikimedia MediaWiki API (no browser needed):

  data/raw/        ← everything (Hampi + other temples) used for LoRA training
  data/reference/  ← intact-gopuram-only shots used as IP-Adapter reference

Run:
    source venv/bin/activate
    python collect_gopuram_images.py
"""

import hashlib, time, sys
import requests
from pathlib import Path
from PIL import Image
import io

ROOT         = Path(__file__).parent
RAW_DIR      = ROOT / "data" / "raw"
REF_DIR      = ROOT / "data" / "reference"
API          = "https://commons.wikimedia.org/w/api.php"
HEADERS      = {"User-Agent": "HampiRevived/2.0 (research project; contact yvg1799@gmail.com)"}
MIN_BYTES    = 80_000   # skip thumbnails
MAX_DIM      = 2000     # cap download size to avoid huge files

RAW_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)


# ─── Wikimedia helpers ─────────────────────────────────────────────────────────

def search_files(query: str, limit: int = 20) -> list[str]:
    """Wikimedia full-text file search — more robust than category membership."""
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}",
        "gsrnamespace": 6,
        "gsrlimit": min(limit * 3, 50),
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
        print(f"    search error ({query}): {e}")
        return []


def category_files(category: str, limit: int = 25) -> list[str]:
    """Return File: titles from a Commons category, falling back to keyword search."""
    cat = category.replace(" ", "_")
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": cat, "cmtype": "file",
        "cmlimit": limit * 2, "format": "json",
    }
    try:
        r = requests.get(API, params=params, headers=HEADERS, timeout=20)
        r.raise_for_status()
        titles = [m["title"] for m in r.json().get("query", {}).get("categorymembers", [])
                  if m.get("title", "").lower().endswith((".jpg", ".jpeg", ".png"))]
        if titles:
            return titles[:limit]
    except Exception as e:
        print(f"    category error: {e}")
    # Fallback: search by keyword
    keyword = cat.split("Category:")[-1].replace("_", " ").replace(",", "")
    return search_files(keyword, limit)


def image_url(title: str, max_width: int = MAX_DIM) -> str | None:
    """Get the thumb URL for a Wikimedia file title."""
    params = {
        "action": "query", "titles": title,
        "prop": "imageinfo", "iiprop": "url|size",
        "iiurlwidth": max_width, "format": "json",
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
        print(f"    URL fetch error: {e}")
    return None


def download_image(url: str, dest: Path) -> bool:
    """Download to dest; skip if already cached or too small."""
    if dest.exists() and dest.stat().st_size > MIN_BYTES:
        print(f"    cached  {dest.name}")
        return True
    try:
        r = requests.get(url, headers=HEADERS, timeout=40, stream=True)
        r.raise_for_status()
        data = b"".join(r.iter_content(8192))
        if len(data) < MIN_BYTES:
            return False
        # Quick PIL sanity check
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


def stem_for(title: str) -> str:
    name = title.split("File:")[-1]
    return hashlib.md5(name.encode()).hexdigest()[:10]


def fetch_batch(category: str, dest_dir: Path, limit: int = 15,
                tag: str = "") -> int:
    """Download up to `limit` images from a category into dest_dir."""
    print(f"\n  [{tag or category.split(':')[-1][:40]}]")
    titles = category_files(category, limit=limit * 2)
    count = 0
    for title in titles:
        if count >= limit:
            break
        url = image_url(title)
        if not url:
            continue
        ext = ".jpg" if ".jpg" in url.lower() or ".jpeg" in url.lower() else ".png"
        dest = dest_dir / (stem_for(title) + ext)
        if download_image(url, dest):
            count += 1
        time.sleep(0.3)
    print(f"  → {count} images saved to {dest_dir.name}/")
    return count


# ─── Search terms ──────────────────────────────────────────────────────────────
# Use keyword search (more reliable than category membership for these temples).

TRAINING_SEARCHES = [
    # More Hampi
    ("Virupaksha temple Hampi gopuram",          RAW_DIR,  8, "Hampi — Virupaksha"),
    ("Vittala temple Hampi stone chariot",        RAW_DIR,  6, "Hampi — Vittala"),
    ("Hazara Rama temple Hampi",                  RAW_DIR,  5, "Hampi — Hazara Rama"),
    ("Hemakuta hill temple Hampi",                RAW_DIR,  4, "Hampi — Hemakuta"),
    ("Hampi ruins gopuram Karnataka",             RAW_DIR,  5, "Hampi — ruins"),
    # Intact South Indian gopurams — teach the model complete-tower proportions
    ("Brihadeeswara temple Thanjavur tower",      RAW_DIR,  7, "Thanjavur Brihadeeswarar"),
    ("Meenakshi temple Madurai gopuram",          RAW_DIR,  7, "Madurai Meenakshi"),
    ("Ranganathaswamy temple Srirangam gopuram",  RAW_DIR,  5, "Srirangam Ranganatha"),
    ("Nataraja temple Chidambaram gopuram",       RAW_DIR,  4, "Chidambaram Nataraja"),
    ("Murudeshwara temple gopuram Karnataka",     RAW_DIR,  4, "Murudeshwara"),
    ("Hoysala temple Karnataka stone carved",     RAW_DIR,  4, "Hoysala stone temples"),
]

# Reference images: intact gopurams for IP-Adapter conditioning.
REFERENCE_SEARCHES = [
    ("Virupaksha temple Hampi tower intact",      REF_DIR,  6, "Ref — Virupaksha intact"),
    ("Brihadeeswara tower Thanjavur stone",       REF_DIR,  5, "Ref — Thanjavur tower"),
    ("Meenakshi gopuram Madurai intact",          REF_DIR,  4, "Ref — Madurai gopuram"),
    ("South Indian temple gopuram intact stone",  REF_DIR,  4, "Ref — generic intact"),
]


# ─── Main ─────────────────────────────────────────────────────────────────────

def fetch_search(query: str, dest_dir: Path, limit: int = 10, tag: str = "") -> int:
    """Search Wikimedia and download up to `limit` results into dest_dir."""
    print(f"\n  [{tag or query[:45]}]")
    titles = search_files(query, limit * 2)
    count  = 0
    for title in titles:
        if count >= limit:
            break
        url = image_url(title)
        if not url:
            continue
        ext  = ".jpg" if any(x in url.lower() for x in (".jpg", ".jpeg")) else ".png"
        dest = dest_dir / (stem_for(title) + ext)
        if download_image(url, dest):
            count += 1
        time.sleep(0.3)
    print(f"  → {count} images saved to {dest_dir.name}/")
    return count


def main():
    print("=" * 62)
    print("Hampi Revived — Gopuram Image Collection")
    print("=" * 62)

    existing_raw = len(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))
    print(f"\nExisting raw images: {existing_raw}")

    # ── 1. Training images ─────────────────────────────────────────────────────
    print("\n── TRAINING IMAGES (data/raw/) ──")
    raw_total = 0
    for query, dest, lim, tag in TRAINING_SEARCHES:
        raw_total += fetch_search(query, dest, lim, tag)

    # ── 2. Reference images ────────────────────────────────────────────────────
    print("\n── REFERENCE IMAGES (data/reference/) ──")
    ref_total = 0
    for query, dest, lim, tag in REFERENCE_SEARCHES:
        ref_total += fetch_search(query, dest, lim, tag)

    # ── 3. Summary ─────────────────────────────────────────────────────────────
    total_raw = len(list(RAW_DIR.glob("*.jpg")) + list(RAW_DIR.glob("*.png")))
    total_ref = len(list(REF_DIR.glob("*.jpg")) + list(REF_DIR.glob("*.png")))

    print(f"\n{'='*62}")
    print(f"Collection complete")
    print(f"  data/raw/       : {total_raw} images  (+{total_raw - existing_raw} new)")
    print(f"  data/reference/ : {total_ref} images")
    print(f"\nNext step:")
    print(f"  FORCE_RETRAIN=1 python lora_pipeline.py")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
