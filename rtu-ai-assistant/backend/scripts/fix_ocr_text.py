"""
fix_ocr_text.py
---------------
Cleans malformed OCR-extracted question text in the SQLite database.

Fixes applied (in order):
  1. Remove artifact exam codes (T-435, P-212, 3E1201, paper codes, Contd...)
  2. Fix hyphenated word-breaks across lines: "vari- ables" => "variables"
  3. Insert space before an uppercase letter that immediately follows a lowercase
     letter (concatenated words), e.g. "Whatis" => "What is"
  4. Normalize multiple whitespace into a single space
  5. Strip leading/trailing whitespace
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "storage" / "rtu_metadata.db"


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

def clean_ocr_text(text: str) -> str:
    """Apply all OCR-correction rules and return the cleaned text."""
    if not text:
        return text

    # 1a. Remove embedded paper/exam reference codes like "3E1201/T-435", "3E1206/2025"
    #     Pattern: alphanumeric code / alphanumeric code
    text = re.sub(r'\b[A-Z0-9]{4,8}/[A-Z]?-?\d{3,4}\b', ' ', text)
    text = re.sub(r'\b[A-Z0-9]{4,8}/\d{4}\b', ' ', text)

    # 1b. Remove standalone artifact exam codes: T-435, P-212, T-48 etc.
    #     Must be surrounded by whitespace / start-end (not part of a word)
    text = re.sub(r'(?<!\w)[A-Z]-\d{2,4}(?!\w)', ' ', text)

    # 1c. Remove standalone alphanumeric paper codes like "3E1201", "4785"
    #     (4-8 uppercase alphanumeric chars that look like paper codes)
    text = re.sub(r'\b[A-Z]\d{4,7}\b', ' ', text)   # e.g. 3E1201
    text = re.sub(r'\b\d{4,5}\b(?=\s*(?:Paper Code|Contd|$))', ' ', text)  # e.g. 4785

    # 1d. Remove "Paper Code XXXXX" labels
    text = re.sub(r'Paper\s+Code\s+[A-Z0-9]{4,8}', ' ', text)

    # 1e. Remove trailing/embedded "(N) Contd..." patterns
    text = re.sub(r'/?\s*\(\d+\)\s+I?Contd\.?\.?\.?', ' ', text)
    text = re.sub(r'\s+I?Contd\.?\.?\.?', ' ', text)

    # 1f. Remove trailing artifacts like "OR 5", "UNIT-III 3(a)", "p UNIT-V 5"
    text = re.sub(r'\s+UNIT-[IVX]+\s+\d+', ' ', text)
    text = re.sub(r'\s+OR\s+\d+$', '', text)
    text = re.sub(r'\s+p\s+UNIT-[IVX]+\s+\d+$', '', text)
    # trailing slash before reference
    text = re.sub(r'\s*/\s*$', '', text)
    text = re.sub(r'\s+/\s+\(\d+\).*$', '', text)

    # 1g. Remove trailing lone digits (page numbers)
    text = re.sub(r'\s+\d+\s*$', '', text)

    # 1h. Remove "10x2=20" type score artifacts at end
    text = re.sub(r'\s*\d+x\d+=\d+\s*$', '', text)

    # 2. Fix hyphenated word-breaks across lines: "vari- ables" => "variables"
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # 3a. Insert space before uppercase+lowercase when immediately after lowercase:
    #     "([a-z])([A-Z][a-z])" — catches "Whatis" (What+is), "Poissondistribution"
    #     Apply multiple times to cascade (e.g. "DefinePoissondistribution")
    for _ in range(5):
        new = re.sub(r'([a-z])([A-Z][a-z])', r'\1 \2', text)
        if new == text:
            break
        text = new

    # 3b. Insert space between lowercase and uppercase followed by space or end
    #     Handles cases like "Whatis " where 'is' is lowercase but 'W' already split
    #     (This catches remaining glued pairs after 3a)
    text = re.sub(r'([a-z]{2,})([A-Z][a-z])', r'\1 \2', text)

    # 4. Normalize multiple whitespace to single space
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # 5. Strip leading / trailing whitespace and punctuation artifacts
    text = text.strip(' \t\n\r·/,')

    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    cursor.execute("SELECT id, text FROM questions")
    rows = cursor.fetchall()

    cleaned_count = 0
    skipped_count = 0
    examples = []  # store (id, original, cleaned) for the first 5 diffs

    updates = []
    for row_id, original_text in rows:
        cleaned = clean_ocr_text(original_text)
        if cleaned != original_text:
            updates.append((cleaned, row_id))
            cleaned_count += 1
            if len(examples) < 5:
                examples.append((row_id, original_text, cleaned))
        else:
            skipped_count += 1

    if updates:
        cursor.executemany("UPDATE questions SET text = ? WHERE id = ?", updates)
        conn.commit()

    conn.close()

    print(f"\n{'='*60}")
    print(f"OCR Text Cleaning Summary")
    print(f"{'='*60}")
    print(f"  Total questions processed : {len(rows)}")
    print(f"  Questions cleaned         : {cleaned_count}")
    print(f"  Questions skipped (clean) : {skipped_count}")
    print(f"\nBefore / After Examples:")
    print(f"{'-'*60}")
    for i, (qid, before, after) in enumerate(examples, 1):
        print(f"\n[Example {i}]  ID: {qid[:50]}")
        print(f"  BEFORE: {before[:200]}")
        print(f"  AFTER : {after[:200]}")
    print(f"\n{'='*60}")
    print("Done. Database updated.\n")

    return cleaned_count, skipped_count, examples


if __name__ == "__main__":
    main()
