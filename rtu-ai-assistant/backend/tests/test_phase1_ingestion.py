"""
Unit Tests for Phase 1: Ingestion & Canonical Schema Validation
"""

import sqlite3
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BACKEND_DIR / "data" / "storage" / "rtu_metadata.db"
MANIFEST_PATH = BACKEND_DIR / "data" / "indexes" / "ingestion_manifest.json"

def test_database_exists_and_populated():
    assert DB_PATH.exists(), f"Database does not exist at {DB_PATH}"
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Check papers count
    c.execute("SELECT COUNT(*) FROM papers")
    paper_count = c.fetchone()[0]
    assert paper_count >= 123, f"Expected at least 123 papers, got {paper_count}"
    
    # Check questions count
    c.execute("SELECT COUNT(*) FROM questions")
    question_count = c.fetchone()[0]
    assert question_count > 300, f"Expected >300 canonical questions, got {question_count}"
    
    # Check subjects count
    c.execute("SELECT COUNT(*) FROM subjects")
    subject_count = c.fetchone()[0]
    assert subject_count >= 12, f"Expected >=12 subjects, got {subject_count}"
    
    conn.close()

def test_canonical_question_schema_integrity():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT q.id, q.paper_id, s.name, s.code, p.year, q.part, q.marks, q.text, p.source_pdf, q.page_number
        FROM questions q
        JOIN papers p ON q.paper_id = p.id
        JOIN subjects s ON p.subject_id = s.id
        WHERE length(trim(q.text)) >= 8
        LIMIT 20
    """)
    rows = c.fetchall()
    assert len(rows) > 0
    
    for r in rows:
        q_id, p_id, sub, code, yr, sec, marks, text, pdf, page = r
        assert q_id.startswith("RTU_"), f"Invalid question_id format: {q_id}"
        assert 2012 <= yr <= 2026, f"Invalid year: {yr}"
        assert len(text.strip()) >= 8, f"Empty or too short question text: {text}"
        assert marks > 0, f"Invalid marks: {marks}"
        assert page >= 1, f"Invalid page number: {page}"
        assert pdf.endswith(".pdf"), f"Invalid PDF filename: {pdf}"
        
    conn.close()

def test_ingestion_manifest_validity():
    assert MANIFEST_PATH.exists(), f"Manifest missing at {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    assert manifest["validation_status"] == "PASSED"
    assert manifest["total_papers_cataloged"] == 123
    assert manifest["total_questions_ingested"] > 300
    assert len(manifest["validation_errors"]) == 0

