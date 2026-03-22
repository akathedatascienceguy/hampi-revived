"""
scrape_hampi_images.py — Use Playwright to browse Wikimedia Commons
and download high-quality real photographs of Hampi.
"""

import os
import time
import hashlib
import requests
from pathlib import Path
from playwright.sync_api import sync_playwright

SAVE_DIR = "data/raw"
TARGET = 20
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}

def download(url, save_dir):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        if "image" not in ctype:
            return None
        ext = ".jpg" if "jpeg" in ctype or "jpg" in url.lower() else ".png"
        fname = hashlib.md5(url.encode()).hexdigest()[:10] + ext
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            return path
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        size = os.path.getsize(path)
        if size < 30_000:          # skip thumbnails < 30 KB
            os.remove(path)
            return None
        print(f"  ✅ {fname}  ({size//1024} KB)")
        return path
    except Exception as e:
        print(f"  ⚠️  {url[:60]}  → {e}")
        return None


def scrape():
    saved = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            viewport={"width": 1400, "height": 900},
        )
        page = ctx.new_page()

        # ── 1. Wikimedia Commons Category ─────────────────────────────────────
        print("\n🔍  Wikimedia Commons: Category:Photographs_of_Hampi")
        page.goto("https://commons.wikimedia.org/wiki/Category:Photographs_of_Hampi", wait_until="networkidle", timeout=30000)
        time.sleep(2)

        # Collect all thumbnail links → click each to get full-res URL
        thumb_links = page.query_selector_all("a.image")
        print(f"  Found {len(thumb_links)} gallery items")

        for link in thumb_links[:30]:
            if len(saved) >= TARGET:
                break
            href = link.get_attribute("href")
            if not href:
                continue
            full_url = "https://commons.wikimedia.org" + href if href.startswith("/") else href
            try:
                detail = ctx.new_page()
                detail.goto(full_url, wait_until="domcontentloaded", timeout=20000)
                # The original file link
                orig = detail.query_selector("a.internal")
                if orig:
                    img_url = orig.get_attribute("href")
                    if img_url and img_url.startswith("//"):
                        img_url = "https:" + img_url
                    if img_url:
                        path = download(img_url, SAVE_DIR)
                        if path:
                            saved.append(path)
                detail.close()
            except Exception as e:
                print(f"  detail page error: {e}")
            time.sleep(0.5)

        # ── 2. Wikimedia Commons: Hampi sub-categories ────────────────────────
        subcats = [
            "Category:Virupaksha_Temple,_Hampi",
            "Category:Vittala_Temple,_Hampi",
            "Category:Lotus_Mahal,_Hampi",
            "Category:Elephant_Stables,_Hampi",
            "Category:Hazara_Rama_Temple,_Hampi",
        ]
        for cat in subcats:
            if len(saved) >= TARGET:
                break
            print(f"\n🔍  {cat}")
            try:
                page.goto(f"https://commons.wikimedia.org/wiki/{cat}", wait_until="networkidle", timeout=25000)
                time.sleep(1.5)
                links = page.query_selector_all("a.image")
                print(f"  Found {len(links)} images")
                for link in links[:10]:
                    if len(saved) >= TARGET:
                        break
                    href = link.get_attribute("href")
                    if not href:
                        continue
                    full_url = "https://commons.wikimedia.org" + href if href.startswith("/") else href
                    try:
                        detail = ctx.new_page()
                        detail.goto(full_url, wait_until="domcontentloaded", timeout=20000)
                        orig = detail.query_selector("a.internal")
                        if orig:
                            img_url = orig.get_attribute("href")
                            if img_url and img_url.startswith("//"):
                                img_url = "https:" + img_url
                            if img_url:
                                path = download(img_url, SAVE_DIR)
                                if path:
                                    saved.append(path)
                        detail.close()
                    except Exception as e:
                        print(f"  detail error: {e}")
                    time.sleep(0.4)
            except Exception as e:
                print(f"  Category error: {e}")

        # ── 3. Wikipedia Hampi article images ─────────────────────────────────
        if len(saved) < TARGET:
            print(f"\n🔍  Wikipedia article images")
            page.goto("https://en.wikipedia.org/wiki/Hampi", wait_until="networkidle", timeout=25000)
            time.sleep(2)
            imgs = page.query_selector_all("img")
            for img in imgs:
                if len(saved) >= TARGET:
                    break
                src = img.get_attribute("src") or ""
                # Get higher-res version by bumping URL
                if "upload.wikimedia.org" in src and src.endswith((".jpg", ".png", ".jpeg")):
                    # Try to get 1200px version
                    src_full = src.replace("//", "https://")
                    # Replace thumb size
                    import re
                    src_full = re.sub(r'/\d+px-', '/1200px-', src_full)
                    path = download(src_full, SAVE_DIR)
                    if path:
                        saved.append(path)
                time.sleep(0.2)

        browser.close()

    print(f"\n✅  Downloaded {len(saved)} images to '{SAVE_DIR}/'")
    for p in saved:
        print(f"   {p}")
    return saved


if __name__ == "__main__":
    scrape()
