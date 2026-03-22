"""
scrape_hampi_v2.py — Playwright-based browser scraper.
Navigates like a real user: Wikimedia Commons gallery → individual file pages
→ saves full-res images to data/raw/
"""

import os, time, re, requests, hashlib
from pathlib import Path
from playwright.sync_api import sync_playwright

SAVE_DIR = "data/raw"
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
TARGET = 20

def save_url(url, label=""):
    try:
        # Use the same session cookies the browser established
        r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Referer": "https://commons.wikimedia.org/",
            "Accept": "image/avif,image/webp,image/apng,*/*",
        }, timeout=30, stream=True)
        r.raise_for_status()
        ct = r.headers.get("content-type","")
        if "image" not in ct and not url.lower().endswith((".jpg",".jpeg",".png")):
            return None
        ext = ".jpg" if "jpeg" in ct or url.lower().endswith((".jpg",".jpeg")) else ".png"
        name = hashlib.md5(url.encode()).hexdigest()[:10] + ext
        path = os.path.join(SAVE_DIR, name)
        if os.path.exists(path) and os.path.getsize(path) > 50_000:
            print(f"  (cached) {label}")
            return path
        data = b"".join(r.iter_content(8192))
        if len(data) < 50_000:
            return None
        with open(path,"wb") as f:
            f.write(data)
        print(f"  ✅ {name}  {len(data)//1024} KB  {label}")
        return path
    except Exception as e:
        print(f"  ❌ {str(e)[:80]}  {label[:40]}")
        return None


def scrape():
    saved = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--disable-blink-features=AutomationControlled"])
        ctx = browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1400, "height": 900},
            locale="en-US",
        )
        page = ctx.new_page()

        # ── STRATEGY 1: browse Commons category pages with real browser ────────
        categories = [
            "https://commons.wikimedia.org/wiki/Hampi",
            "https://commons.wikimedia.org/wiki/Category:Virupaksha_Temple,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Vittala_Temple,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Elephant_Stables,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Lotus_Mahal,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Hazara_Rama_Temple,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Stone_Chariot,_Hampi",
            "https://commons.wikimedia.org/wiki/Category:Hampi_Bazaar",
        ]

        for cat_url in categories:
            if len(saved) >= TARGET:
                break
            print(f"\n🌐  {cat_url.split('/')[-1].replace('_',' ')}")
            try:
                page.goto(cat_url, wait_until="networkidle", timeout=30000)
                time.sleep(2)

                # Collect file page links
                links = page.eval_on_selector_all(
                    "a.image, li.gallerybox a, .thumb a",
                    "els => els.map(e => e.href)"
                )
                links = list(dict.fromkeys(l for l in links if l))  # dedupe
                print(f"  {len(links)} gallery links found")

                for href in links:
                    if len(saved) >= TARGET:
                        break
                    if "/wiki/File:" not in href:
                        continue
                    try:
                        page.goto(href, wait_until="domcontentloaded", timeout=20000)
                        time.sleep(0.8)

                        # Get the original full-res file URL
                        orig = page.query_selector("a.internal, #file a, .fullImageLink a")
                        if orig:
                            img_url = orig.get_attribute("href") or ""
                            if img_url.startswith("//"):
                                img_url = "https:" + img_url
                            if img_url:
                                fname = href.split("File:")[-1]
                                path = save_url(img_url, fname[:50])
                                if path:
                                    saved.append(path)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"  file page error: {e}")
            except Exception as e:
                print(f"  cat error: {e}")

        # ── STRATEGY 2: Wikimedia image search ─────────────────────────────────
        if len(saved) < TARGET:
            print(f"\n🌐  Wikimedia image search")
            search_queries = [
                "https://commons.wikimedia.org/w/index.php?search=hampi+temple&ns6=1&title=Special:Search&searchToken=xxx",
                "https://commons.wikimedia.org/w/index.php?search=virupaksha+hampi&ns6=1&title=Special:Search",
                "https://commons.wikimedia.org/w/index.php?search=vittala+hampi+ruins&ns6=1&title=Special:Search",
            ]
            for sq in search_queries:
                if len(saved) >= TARGET:
                    break
                try:
                    page.goto(sq, wait_until="networkidle", timeout=25000)
                    time.sleep(2)
                    links = page.eval_on_selector_all(
                        ".searchresults a, .mw-search-result-heading a",
                        "els => els.map(e => e.href)"
                    )
                    for href in links[:12]:
                        if len(saved) >= TARGET:
                            break
                        if "/wiki/File:" not in href:
                            continue
                        try:
                            page.goto(href, wait_until="domcontentloaded", timeout=20000)
                            time.sleep(0.6)
                            orig = page.query_selector("a.internal, #file a, .fullImageLink a")
                            if orig:
                                img_url = orig.get_attribute("href") or ""
                                if img_url.startswith("//"):
                                    img_url = "https:" + img_url
                                if img_url:
                                    path = save_url(img_url, href.split("File:")[-1][:50])
                                    if path:
                                        saved.append(path)
                            time.sleep(0.5)
                        except Exception as e:
                            print(f"  {e}")
                except Exception as e:
                    print(f"  search error: {e}")

        # ── STRATEGY 3: Unsplash search (free, no login needed) ────────────────
        if len(saved) < TARGET:
            print(f"\n🌐  Trying Unsplash for Hampi / Karnataka temple images")
            try:
                page.goto("https://unsplash.com/s/photos/hampi", wait_until="networkidle", timeout=30000)
                time.sleep(3)
                # Scroll to load more
                for _ in range(3):
                    page.keyboard.press("End")
                    time.sleep(1.5)

                imgs = page.eval_on_selector_all(
                    "img[srcset], figure img",
                    """els => els.map(e => {
                        let s = e.srcset || e.src || '';
                        // pick highest res from srcset
                        let parts = s.split(',').map(x => x.trim().split(' '));
                        let best = parts.reduce((a,b) => {
                            let aw = parseInt((a[1]||'0w'));
                            let bw = parseInt((b[1]||'0w'));
                            return bw > aw ? b : a;
                        }, parts[0]);
                        return best ? best[0] : e.src;
                    })"""
                )
                print(f"  {len(imgs)} images found")
                for url in imgs:
                    if len(saved) >= TARGET:
                        break
                    if not url or "photo" not in url:
                        continue
                    # Get larger version
                    url = re.sub(r'w=\d+', 'w=1200', url)
                    url = re.sub(r'&q=\d+', '&q=85', url)
                    path = save_url(url, url[-40:])
                    if path:
                        saved.append(path)
                    time.sleep(0.3)
            except Exception as e:
                print(f"  Unsplash error: {e}")

        browser.close()

    print(f"\n{'='*50}")
    print(f"✅  Total downloaded: {len(saved)} images")
    for s in saved:
        size = os.path.getsize(s)
        print(f"   {s}  ({size//1024} KB)")
    return saved


if __name__ == "__main__":
    # Clear old synthetic images
    for f in Path(SAVE_DIR).glob("synthetic_*"):
        f.unlink()
    scrape()
