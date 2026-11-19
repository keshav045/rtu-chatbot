"""
Batch Extraction Engine for RTU Question Papers
Extracts individual exam questions and raw OCR pages for all unextracted papers
using RapidOCR + PyMuPDF and writes canonical JSON files to cleaned_dataset/extracted_text/.
"""

import sys
import os
import re
import json
import time
from pathlib import Path
import pandas as pd
import fitz
from rapidocr_onnxruntime import RapidOCR

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
CLEAN_DIR = BASE_DIR / "cleaned_dataset"
CATALOG_PATH = CLEAN_DIR / "metadata" / "dataset_catalog.csv"
EXTRACTED_DIR = CLEAN_DIR / "extracted_text"
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OCR Cleaning Logic (mirrors fix_ocr_text.py)
# ---------------------------------------------------------------------------

def clean_ocr_line(text: str) -> str:
    if not text:
        return ""
    # Remove watermarks
    text = re.sub(r'ersahilkagyan\.com', '', text, flags=re.I)
    text = re.sub(r'Free RTU papers at\s*', '', text, flags=re.I)
    text = re.sub(r'rtuonline\.com', '', text, flags=re.I)
    text = re.sub(r'Contd\.?\.?\.?', '', text, flags=re.I)
    # Remove exam reference codes
    text = re.sub(r'\b[A-Z0-9]{4,8}/[A-Z]?-?\d{3,4}\b', ' ', text)
    text = re.sub(r'(?<!\w)[A-Z]-\d{2,4}(?!\w)', ' ', text)
    # Fix hyphenated breaks
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    # Fix glued words
    for _ in range(3):
        new = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        if new == text:
            break
        text = new
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------------------------------------------------------------------
# Question Parsing Logic
# ---------------------------------------------------------------------------

def parse_pages_to_questions(pages_raw, metadata):
    questions = []
    current_part = "Part A"
    current_unit = "Unit 1"
    
    part_a_re = re.compile(r'PART\s*[-–—]?\s*A\b', re.I)
    part_b_re = re.compile(r'PART\s*[-–—]?\s*B\b', re.I)
    part_c_re = re.compile(r'PART\s*[-–—]?\s*C\b', re.I)
    unit_re = re.compile(r'UNIT\s*[-–—]?\s*([IVX1-5]+)\b', re.I)
    
    q_start_re = re.compile(r'^(?:Q(?:uestion)?\.?\s*(\d+)|(\d+)\s*[\.\)]|\(([a-h])\))\s*(.*)', re.I)
    header_skip_re = re.compile(
        r'Roll No|Total No|Semester|Examination|Maximum Marks|Time:|Instructions|'
        r'Attempt all|Attempt any|Schematic diagrams|Units of quantities|'
        r'Use of following|carrying marks|passing marks|b\.tech|Min\. Passing',
        re.I
    )

    buffer_q = ""
    buffer_part = current_part
    buffer_page = 1
    
    def flush():
        nonlocal buffer_q, buffer_part, buffer_page
        text = buffer_q.strip()
        text = re.sub(r'\s+', ' ', text)
        # Strip trailing pagination or marks artifacts
        text = text.strip(' \t\n\r·/,')
        if len(text) >= 15:
            # Default marks
            if "part a" in buffer_part.lower():
                marks = 2
            elif "part b" in buffer_part.lower():
                marks = 8
            elif "part c" in buffer_part.lower():
                marks = 15
            else:
                marks = 8
            
            # Explicit marks match at end, e.g. (10) or [8] or (2)
            marks_match = re.search(r'[\(\[]\s*(\d{1,2})\s*[\)\]]\s*$', text)
            if marks_match:
                marks = int(marks_match.group(1))
                text = text[:marks_match.start()].strip()
            
            clean_year = min(int(metadata.get("year", 2024)), 2024)
            questions.append({
                "paper_id": metadata["paper_id"],
                "subject": metadata["subject_short"],
                "subject_name": metadata["subject_name"],
                "subject_code": metadata["subject_code"],
                "semester": metadata["semester"],
                "year": clean_year,
                "exam_session": metadata.get("exam_session", "Main"),
                "part": buffer_part,
                "unit": current_unit,
                "marks": marks,
                "question_text": text
            })
        buffer_q = ""

    for p_info in pages_raw:
        pno = p_info["page"]
        lines = [l["text"] for l in p_info["lines"]]
        
        for line in lines:
            line_str = clean_ocr_line(line)
            if not line_str:
                continue
                
            if part_a_re.search(line_str):
                flush()
                current_part = "Part A"
                continue
            elif part_b_re.search(line_str):
                flush()
                current_part = "Part B"
                continue
            elif part_c_re.search(line_str):
                flush()
                current_part = "Part C"
                continue
            
            u_match = unit_re.search(line_str)
            if u_match:
                current_unit = f"Unit {u_match.group(1)}"
            
            if header_skip_re.search(line_str):
                continue
            
            q_match = q_start_re.match(line_str)
            if q_match:
                flush()
                buffer_part = current_part
                buffer_page = pno
                rest = q_match.group(4) if len(q_match.groups()) >= 4 else ""
                buffer_q = rest
            else:
                if buffer_q:
                    buffer_q += " " + line_str
    
    flush()
    return questions

# ---------------------------------------------------------------------------
# Main Extraction Loop
# ---------------------------------------------------------------------------

def run_batch_extraction():
    print("=" * 70)
    print("RTU PAST PAPERS BATCH EXTRACTION ENGINE (RAPIDOCR)")
    print("=" * 70)

    if not CATALOG_PATH.exists():
        print(f"Error: Catalog not found at {CATALOG_PATH}")
        return

    df = pd.read_csv(CATALOG_PATH)
    total_catalog = len(df)
    
    # Identify unextracted papers
    already_extracted = set()
    for f in EXTRACTED_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fl:
                d = json.load(fl)
                if len(d.get("questions", [])) > 0:
                    already_extracted.add(f.stem)
        except Exception:
            pass

    to_extract = df[~df["paper_id"].isin(already_extracted)]
    print(f"Total Papers in Catalog : {total_catalog}")
    print(f"Already Extracted        : {len(already_extracted)}")
    print(f"Papers to Extract        : {len(to_extract)}")
    print("=" * 70)

    if len(to_extract) == 0:
        print("All papers have already been extracted. Nothing to do.")
        return

    engine = RapidOCR()
    extracted_count = 0
    total_questions_new = 0
    t0_start = time.time()

    for idx, (_, row) in enumerate(to_extract.iterrows(), 1):
        pid = row["paper_id"]
        rel_path = row["clean_relative_path"]
        pdf_path = CLEAN_DIR / rel_path
        
        t0 = time.time()
        if not pdf_path.exists():
            print(f"[{idx}/{len(to_extract)}] Skipping {pid}: File not found ({pdf_path.name})")
            continue

        try:
            doc = fitz.open(pdf_path)
            pages_raw = []
            
            for pno in range(len(doc)):
                page = doc[pno]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                
                result, _ = engine(img_bytes)
                lines = []
                full_page_text_parts = []
                if result:
                    for lno, line_tuple in enumerate(result, 1):
                        txt = line_tuple[1].strip()
                        lines.append({"line_num": lno, "text": txt})
                        full_page_text_parts.append(txt)
                
                pages_raw.append({
                    "page": pno + 1,
                    "lines": lines,
                    "full_page_text": "\n".join(full_page_text_parts)
                })
            doc.close()

            meta = row.to_dict()
            meta["year"] = min(int(meta.get("year", 2024)), 2024)
            questions = parse_pages_to_questions(pages_raw, meta)
            
            # Save canonical JSON
            out_obj = {
                "metadata": meta,
                "total_pages": len(pages_raw),
                "total_questions_extracted": len(questions),
                "questions": questions,
                "pages_raw": pages_raw
            }
            
            out_file = EXTRACTED_DIR / f"{pid}.json"
            with open(out_file, "w", encoding="utf-8") as out_f:
                json.dump(out_obj, out_f, indent=2, ensure_ascii=False)

            elapsed = time.time() - t0
            extracted_count += 1
            total_questions_new += len(questions)
            print(f"[{idx}/{len(to_extract)}] {pid} ({meta['subject_short']} {meta['year']}) -> {len(questions)} questions ({elapsed:.1f}s)")

        except Exception as e:
            print(f"[{idx}/{len(to_extract)}] ERROR processing {pid}: {e}")

    total_time = time.time() - t0_start
    print("=" * 70)
    print(f"BATCH EXTRACTION COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Papers Extracted : {extracted_count}")
    print(f"Questions Added  : {total_questions_new}")
    print("=" * 70)

if __name__ == "__main__":
    run_batch_extraction()
