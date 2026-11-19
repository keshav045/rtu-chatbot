"""
RTUOnline Question Paper Downloader for Semesters 3 to 8 (CS, AI, DS)
Crawls https://www.rtuonline.com/btech-cs-question-papers.html
Downloads papers for Semesters 3-8 into cleaned_dataset/question_papers/
"""

import sys
import os
import re
import hashlib
import time
import urllib.request
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DIR = BASE_DIR / "cleaned_dataset"
PAPERS_DIR = CLEAN_DIR / "question_papers"
CATALOG_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def fetch_url(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def download_file(url: str, dest_path: Path) -> bool:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
        if len(data) < 1000 or not data.startswith(b"%PDF"):
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    Download error for {url}: {e}")
        return False

def crawl_and_download(target_semesters=None, max_per_sem=50):
    if target_semesters is None:
        target_semesters = [3, 4, 5, 6, 7, 8]

    print("=" * 70)
    print("RTUONLINE.COM CRAWLER & DOWNLOADER (SEMESTERS 3 TO 8: CS, AI, DS)")
    print("=" * 70)

    main_url = "https://www.rtuonline.com/btech-cs-question-papers.html"
    print(f"Fetching index from {main_url}...")
    html = fetch_url(main_url)

    # Find paper links
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+\.html)["\'][^>]*>(.*?)</a>', re.I | re.DOTALL)
    matches = pattern.findall(html)
    print(f"Found {len(matches)} potential paper links on main page.")

    # Filter for target semesters and relevant CS / AI / DS branches
    catalog_df = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
    existing_hashes = set(catalog_df["md5_hash"].dropna().tolist()) if "md5_hash" in catalog_df else set()
    existing_filenames = set(catalog_df["clean_filename"].dropna().tolist()) if "clean_filename" in catalog_df else set()

    sem_counters = {s: 0 for s in target_semesters}
    new_downloads = 0

    for href, raw_text in matches:
        href_clean = href.strip()
        if not href_clean.startswith("http"):
            paper_page_url = f"https://www.rtuonline.com/{href_clean.lstrip('/')}"
        else:
            paper_page_url = href_clean

        # Detect semester from text or href (e.g., "3-sem", "4-sem", "sem-5", "6-sem")
        sem_match = re.search(r'(\d)\s*[-_]?\s*sem', href_clean + " " + raw_text, re.I)
        if not sem_match:
            continue
        sem_num = int(sem_match.group(1))
        if sem_num not in target_semesters:
            continue

        if sem_counters[sem_num] >= max_per_sem:
            continue

        # Paper slug
        slug = Path(href_clean).stem
        pdf_filename = f"{slug}.pdf"

        if pdf_filename in existing_filenames:
            continue

        # Direct PDF URL convention on rtuonline: https://www.rtuonline.com/papers/{slug}.pdf
        pdf_url = f"https://www.rtuonline.com/papers/{pdf_filename}"
        
        # Determine target folder
        sem_folder = PAPERS_DIR / f"Semester_{sem_num}"
        dest_path = sem_folder / "Downloaded" / pdf_filename

        print(f"\n[Sem {sem_num}] Downloading: {pdf_filename}...")
        success = download_file(pdf_url, dest_path)
        if success:
            with open(dest_path, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            
            if md5 in existing_hashes:
                print(f"    Duplicate hash {md5[:8]}, removing.")
                dest_path.unlink()
                continue

            existing_hashes.add(md5)
            existing_filenames.add(pdf_filename)
            sem_counters[sem_num] += 1
            new_downloads += 1
            file_size_kb = round(dest_path.stat().st_size / 1024.0, 1)
            print(f"    Saved: {dest_path.name} ({file_size_kb} KB)")
        else:
            # Fallback: visit the page to scrape the download button
            try:
                page_html = fetch_url(paper_page_url, timeout=10)
                pdf_match = re.search(r'href=["\']([^"\']+\.pdf)["\']', page_html, re.I)
                if pdf_match:
                    found_pdf = pdf_match.group(1).lstrip("/")
                    if not found_pdf.startswith("http"):
                        found_pdf = f"https://www.rtuonline.com/papers/{found_pdf}"
                    success = download_file(found_pdf, dest_path)
                    if success:
                        sem_counters[sem_num] += 1
                        new_downloads += 1
                        print(f"    Saved via page scrape: {dest_path.name}")
            except Exception as e:
                pass

        time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE: {new_downloads} new papers downloaded.")
    for s, count in sem_counters.items():
        print(f"  Semester {s}: {count} papers")
    print("=" * 70)

if __name__ == "__main__":
    crawl_and_download()
