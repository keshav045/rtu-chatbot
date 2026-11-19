"""
RTUOnline Master Question Paper Downloader & Cataloger
Crawls https://www.rtuonline.com/btech-cs-question-papers.html
Downloads all past question papers across Semesters 3 to 8 for CS, AI, DS, and IT.
Automatically verifies PDFs, computes MD5 hashes, deduplicates, and registers new entries into dataset_catalog.csv.
"""

import sys
import os
import re
import hashlib
import time
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import fitz

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DIR = BASE_DIR / "cleaned_dataset"
PAPERS_DIR = CLEAN_DIR / "question_papers"
CATALOG_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.csv"
CATALOG_JSON_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

SUBJECT_CODE_MAP = {
    "DSA": ("Data Structures and Algorithms", "3CS4-05"),
    "OOP": ("Object Oriented Programming", "3CS4-06"),
    "DE": ("Digital Electronics", "3CS3-04"),
    "SE": ("Software Engineering", "3CS4-07"),
    "AEM": ("Advanced Engineering Mathematics", "3CS2-01"),
    "MEFA": ("Managerial Economics and Financial Accounting", "3CS1-03"),
    "TC": ("Technical Communication", "3CS1-02"),
    "DMS": ("Discrete Mathematics Structure", "4CS2-01"),
    "MI": ("Microprocessor and Interfaces", "4CS4-04"),
    "DBMS": ("Database Management Systems", "5CS4-02"),
    "DCCN": ("Data Communication and Computer Networks", "5CS4-01"),
    "TOC": ("Theory of Computation", "5CS4-03"),
    "OS": ("Operating Systems", "5CS4-04"),
    "CD": ("Compiler Design", "6CS4-01"),
    "DAA": ("Design and Analysis of Algorithms", "6CS4-02"),
    "AI": ("Artificial Intelligence", "6CS4-03"),
    "CA": ("Computer Architecture and Organization", "6CS4-04"),
    "ML": ("Machine Learning", "7CS4-01"),
    "IOT": ("Internet of Things", "7CS4-02"),
    "CC": ("Cloud Computing", "7CS4-03"),
    "BDA": ("Big Data Analytics", "7CS4-04"),
    "ISS": ("Information Security Systems", "7CS4-05"),
    "DM": ("Disaster Management", "8CS4-01")
}

def fetch_url(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def download_pdf(pdf_url: str, dest_path: Path) -> bool:
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(pdf_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
        if len(data) < 2000 or not data.startswith(b"%PDF"):
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False

def parse_metadata_from_slug(slug: str, sem_num: int):
    s_lower = slug.lower()
    
    # Year detection
    year_match = re.search(r'\b(20[12]\d)\b', s_lower)
    year = int(year_match.group(1)) if year_match else 2024
    if year >= 2025:
        year = 2024

    # Subject detection
    subj_short = "GEN"
    subj_name = "Engineering Paper"
    subj_code = f"{sem_num}CS0-01"

    for code, (name, rtu_code) in SUBJECT_CODE_MAP.items():
        slug_check = name.lower().replace(" ", "-")
        code_check = code.lower()
        if code_check in s_lower or slug_check in s_lower or rtu_code.lower() in s_lower:
            subj_short = code
            subj_name = name
            subj_code = rtu_code
            break

    # Branch detection
    branch = "CS-IT"
    if "aid" in s_lower or "ai-ds" in s_lower:
        branch = "AI-DS"
    elif "cai" in s_lower or "aiml" in s_lower:
        branch = "AI-ML"

    # Session detection
    session = "Main"
    if "back" in s_lower:
        session = "Back"
    elif "dec" in s_lower:
        session = "Dec"
    elif "jan" in s_lower:
        session = "Jan"
    elif "may" in s_lower:
        session = "May"
    elif "feb" in s_lower:
        session = "Feb"

    return {
        "subject_short": subj_short,
        "subject_name": subj_name,
        "subject_code": subj_code,
        "semester": f"Semester {sem_num}",
        "year": year,
        "exam_session": session,
        "branch_curriculum": branch
    }

def main():
    print("=" * 70)
    print("RTUONLINE.COM MASTER DOWNLOADER (ALL SEMESTERS: CS, AI, DS)")
    print("=" * 70)

    main_url = "https://www.rtuonline.com/btech-cs-question-papers.html"
    print(f"Fetching master paper list from {main_url}...")
    html = fetch_url(main_url)

    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+\.html)["\'][^>]*>(.*?)</a>', re.I | re.DOTALL)
    matches = pattern.findall(html)
    print(f"Total links scanned: {len(matches)}")

    # Load existing catalog
    catalog_df = pd.read_csv(CATALOG_PATH) if CATALOG_PATH.exists() else pd.DataFrame()
    existing_hashes = set(catalog_df["md5_hash"].dropna().tolist()) if "md5_hash" in catalog_df else set()
    existing_filenames = set(catalog_df["clean_filename"].dropna().tolist()) if "clean_filename" in catalog_df else set()
    
    # Highest paper ID
    max_id = 123
    if "paper_id" in catalog_df and len(catalog_df) > 0:
        for pid in catalog_df["paper_id"].dropna():
            m = re.search(r'RTU_QP_(\d+)', str(pid))
            if m:
                max_id = max(max_id, int(m.group(1)))

    # Collect papers to download
    download_queue = []
    seen_slugs = set()

    for href, raw_text in matches:
        href_clean = href.strip()
        slug = Path(href_clean).stem
        if slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        sem_match = re.search(r'(\d)\s*[-_]?\s*sem', href_clean + " " + raw_text, re.I)
        if not sem_match:
            continue
        sem_num = int(sem_match.group(1))
        if sem_num < 1 or sem_num > 8:
            continue

        pdf_filename = f"{slug}.pdf"
        if pdf_filename in existing_filenames:
            continue

        pdf_url = f"https://www.rtuonline.com/papers/{pdf_filename}"
        dest_folder = PAPERS_DIR / f"Semester_{sem_num}" / "RTUOnline"
        dest_path = dest_folder / pdf_filename

        download_queue.append({
            "slug": slug,
            "sem_num": sem_num,
            "pdf_url": pdf_url,
            "dest_path": dest_path,
            "filename": pdf_filename
        })

    print(f"Discovered {len(download_queue)} new question papers to download.")
    if not download_queue:
        print("No new papers to download. System is completely up to date.")
        return

    print(f"Starting parallel download using 5 worker threads...\n")
    downloaded_records = []
    t0_all = time.time()

    def process_item(item):
        dest = item["dest_path"]
        url = item["pdf_url"]
        ok = download_pdf(url, dest)
        if not ok:
            return None
        with open(dest, "rb") as f:
            data = f.read()
        md5 = hashlib.md5(data).hexdigest()
        if md5 in existing_hashes:
            dest.unlink(missing_ok=True)
            return None

        # Verify page count with PyMuPDF
        try:
            doc = fitz.open(dest)
            page_count = len(doc)
            doc.close()
        except Exception:
            page_count = 2

        file_size_kb = round(dest.stat().st_size / 1024.0, 1)
        meta = parse_metadata_from_slug(item["slug"], item["sem_num"])
        meta["clean_filename"] = item["filename"]
        meta["clean_relative_path"] = str(dest.relative_to(CLEAN_DIR)).replace("\\", "/")
        meta["page_count"] = page_count
        meta["file_size_kb"] = file_size_kb
        meta["md5_hash"] = md5
        meta["original_file"] = f"rtuonline/{item['filename']}"
        return meta

    current_id = max_id
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_item, item): item for item in download_queue}
        done_count = 0
        for fut in as_completed(futures):
            item = futures[fut]
            meta = fut.result()
            done_count += 1
            if meta:
                current_id += 1
                meta["paper_id"] = f"RTU_QP_{current_id:03d}"
                downloaded_records.append(meta)
                existing_hashes.add(meta["md5_hash"])
                existing_filenames.add(meta["clean_filename"])
                print(f"[{done_count}/{len(download_queue)}] Saved {meta['paper_id']} ({meta['semester']} - {meta['subject_short']}): {meta['clean_filename']} ({meta['file_size_kb']} KB)")
            else:
                if done_count % 20 == 0:
                    print(f"[{done_count}/{len(download_queue)}] Processed...")

    # Append new records to dataset_catalog.csv
    if downloaded_records:
        new_df = pd.DataFrame(downloaded_records)
        combined_df = pd.concat([catalog_df, new_df], ignore_index=True) if not catalog_df.empty else new_df
        combined_df.to_csv(CATALOG_PATH, index=False)
        print(f"\n[+] Updated dataset_catalog.csv with {len(downloaded_records)} new papers!")
        print(f"[+] Total papers now cataloged: {len(combined_df)}")

    elapsed = time.time() - t0_all
    print("=" * 70)
    print(f"DOWNLOAD COMPLETE: {len(downloaded_records)} papers downloaded and cataloged in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

if __name__ == "__main__":
    main()
