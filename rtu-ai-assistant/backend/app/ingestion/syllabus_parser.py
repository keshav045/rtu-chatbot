"""
Syllabus Parser and Curriculum Ingestion Module for RTU Exam Helper Chatbot
Implements Section 5 & 7 of Implementation Plan Version 2.
Parses RTU syllabus PDFs, creates universities, schemes, branches, subjects, units,
and topics with estimated study times.
"""

import sys
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
SYLLABUS_DIR = PROJECT_ROOT / "syllabus"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal, init_db, engine
from app.database.models import (
    Base, University, Scheme, Branch, Subject, SubjectOffering, Unit, Topic, Paper, Question
)

# Standard subject mapping for RTU CS/IT Curriculum
CORE_SUBJECT_MAP = {
    "AEM": {"name": "Advanced Engineering Mathematics", "code": "3CS2-01", "sem": 3},
    "DE": {"name": "Digital Electronics", "code": "3CS3-04", "sem": 3},
    "DSA": {"name": "Data Structures and Algorithms", "code": "3CS4-05", "sem": 3},
    "OOP": {"name": "Object Oriented Programming", "code": "3CS4-06", "sem": 3},
    "SE": {"name": "Software Engineering", "code": "3CS4-07", "sem": 3},
    "DMS": {"name": "Discrete Mathematics Structure", "code": "4CS2-01", "sem": 4},
    "TC": {"name": "Technical Communication", "code": "4CS1-02", "sem": 4},
    "MEFA": {"name": "Managerial Economics and Financial Accounting", "code": "3CS1-03", "sem": 3},
    "DBMS": {"name": "Database Management System", "code": "4CS4-05", "sem": 4},
    "TOC": {"name": "Theory of Computation", "code": "4CS4-06", "sem": 4},
    "DCCN": {"name": "Data Communication and Computer Networks", "code": "4CS4-07", "sem": 4},
    "POC": {"name": "Principles of Communication", "code": "4CS3-04", "sem": 4},
    "MI": {"name": "Microprocessor And Interfaces", "code": "5CS3-01", "sem": 5},
    "OS": {"name": "Operating System", "code": "5CS4-03", "sem": 5},
    "AOA": {"name": "Analysis of Algorithms", "code": "5CS4-05", "sem": 5},
    "CN": {"name": "Computer Networks", "code": "5CS4-01", "sem": 5},
}

def seed_foundational_metadata(db):
    """Seeds universities, schemes, and branches."""
    print("[SyllabusParser] Seeding foundational universities and schemes...")
    
    # University
    uni = db.query(University).filter(University.id == "RTU").first()
    if not uni:
        uni = University(id="RTU", name="Rajasthan Technical University, Kota")
        db.add(uni)

    # Scheme
    scheme = db.query(Scheme).filter(Scheme.id == "SCHEME_RTU_2018").first()
    if not scheme:
        scheme = Scheme(id="SCHEME_RTU_2018", university_id="RTU", name="RTU B.Tech CBCS Scheme (2018 Onwards)", year_start=2018)
        db.add(scheme)

    # Branches
    branches_data = [
        ("CSE", "Computer Science & Engineering"),
        ("IT", "Information Technology"),
        ("ECE", "Electronics & Communication Engineering"),
        ("EE", "Electrical Engineering"),
        ("ME", "Mechanical Engineering"),
        ("CE", "Civil Engineering")
    ]
    for code, bname in branches_data:
        b = db.query(Branch).filter(Branch.code == code).first()
        if not b:
            b = Branch(id=f"BRANCH_{code}", code=code, name=bname)
            db.add(b)

    db.commit()

def parse_syllabus_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extracts subjects, units, and topics from an RTU syllabus PDF."""
    if not pdf_path.exists():
        print(f"[SyllabusParser] Warning: Syllabus PDF not found at {pdf_path}")
        return []

    doc = fitz.open(pdf_path)
    subjects_extracted = []

    for pno in range(len(doc)):
        page = doc[pno]
        blocks = page.get_text("blocks")
        
        page_subject_code = None
        page_subject_name = None
        units_found = []
        
        for b in blocks:
            txt = b[4].strip()
            
            # Check for subject header e.g. "3IT4-05: Data Structures and Algorithms"
            header_match = re.search(r"(\d[A-Z]{2}\d-\d{2})\s*:\s*([^\n\r]+)", txt)
            if header_match and not any(k in txt.lower() for k in ["lab", "scheme", "hours", "credit", "teaching"]):
                code = header_match.group(1).strip()
                name = header_match.group(2).strip()
                page_subject_code = code
                page_subject_name = name

            # Check for unit blocks starting with a digit 1-5 e.g. "1 Stacks: Basic Stack Operations..."
            unit_match = re.match(r"^([1-5])\s+([A-Za-z0-9].*)", txt, re.DOTALL)
            if unit_match:
                unit_no = int(unit_match.group(1))
                unit_text = unit_match.group(2).strip()
                # Remove trailing hour numbers
                unit_clean = re.sub(r"\s+\d+\s*$", "", unit_text).strip()
                units_found.append((unit_no, unit_clean))

        if page_subject_code and units_found:
            subjects_extracted.append({
                "code": page_subject_code,
                "name": page_subject_name,
                "page": pno + 1,
                "units": units_found
            })

    doc.close()
    return subjects_extracted

def extract_topics_from_unit_text(unit_text: str) -> List[Tuple[str, int]]:
    """Splits unit content into clean topic concepts with estimated study minutes."""
    # Look for major sub-headers or split by colon/semicolon/full stop
    lines = unit_text.replace("\n", " ").split(".")
    raw_topics = []
    for l in lines:
        parts = re.split(r"[:;]", l)
        for p in parts:
            clean = p.strip()
            if len(clean) > 4 and len(clean) < 90 and not re.match(r"^\d+$", clean):
                raw_topics.append(clean)

    if not raw_topics:
        raw_topics = [unit_text[:70].strip()]

    # Assign estimated study minutes: shorter concepts 30 mins, complex concepts 60 mins
    topics_with_time = []
    for t in raw_topics:
        words = len(t.split())
        mins = 30 if words <= 4 else (45 if words <= 8 else 60)
        topics_with_time.append((t, mins))

    return topics_with_time

def ingest_curriculum_and_syllabus(db):
    """Parses syllabus and populates subjects, offerings, units, and topics."""
    seed_foundational_metadata(db)

    # 1. Register Core Subjects
    print("\n[SyllabusParser] Registering core subjects taxonomy...")
    for short_code, info in CORE_SUBJECT_MAP.items():
        subj_id = f"SUBJ_{short_code}"
        subj = db.query(Subject).filter(Subject.id == subj_id).first()
        if not subj:
            subj = Subject(
                id=subj_id,
                code=info["code"],
                name=info["name"],
                short_name=short_code,
                data_tier="Tier C"  # updated after paper tagging
            )
            db.add(subj)
            db.flush()

            # Offering for CSE and IT
            for bcode in ["CSE", "IT"]:
                offering_id = f"OFFER_{short_code}_{bcode}"
                off = db.query(SubjectOffering).filter(SubjectOffering.id == offering_id).first()
                if not off:
                    off = SubjectOffering(
                        id=offering_id,
                        subject_id=subj_id,
                        branch_id=f"BRANCH_{bcode}",
                        semester=info["sem"],
                        scheme_id="SCHEME_RTU_2018"
                    )
                    db.add(off)

    db.commit()

    # 2. Parse syllabus PDF
    cs_it_pdf = SYLLABUS_DIR / "Syllabus_2018onwards.pdf"
    print(f"\n[SyllabusParser] Parsing syllabus from: {cs_it_pdf}")
    extracted = parse_syllabus_pdf(cs_it_pdf)
    print(f"[SyllabusParser] Extracted syllabus content for {len(extracted)} course modules.")

    total_units_created = 0
    total_topics_created = 0

    # Build map from code / name to short code
    code_to_short = {}
    for short, info in CORE_SUBJECT_MAP.items():
        code_to_short[info["code"]] = short
        # Also match base e.g. "3IT4-05" or "3CS4-05"
        base_match = info["code"][3:] # "4-05"
        code_to_short[base_match] = short

    for item in extracted:
        code = item["code"]
        name = item["name"]
        
        # Determine subject short code
        matched_short = None
        for s_code, s_info in CORE_SUBJECT_MAP.items():
            if s_info["name"].lower() in name.lower() or name.lower() in s_info["name"].lower():
                matched_short = s_code
                break
            if s_info["code"][3:] in code:
                matched_short = s_code
                break

        if not matched_short:
            continue

        subj_id = f"SUBJ_{matched_short}"
        for unit_no, unit_content in item["units"]:
            unit_id = f"UNIT_{matched_short}_U{unit_no}"
            unit = db.query(Unit).filter(Unit.id == unit_id).first()
            if not unit:
                # First line or first 50 chars as title
                title_line = unit_content.split(":")[0].strip()[:100]
                unit = Unit(
                    id=unit_id,
                    subject_id=subj_id,
                    scheme_id="SCHEME_RTU_2018",
                    unit_no=unit_no,
                    title=f"Unit {unit_no}: {title_line}"
                )
                db.add(unit)
                db.flush()
                total_units_created += 1

            # Extract constituent topics
            topic_items = extract_topics_from_unit_text(unit_content)
            for t_idx, (topic_name, est_mins) in enumerate(topic_items, 1):
                topic_id = f"TOPIC_{matched_short}_U{unit_no}_{t_idx:02d}"
                topic = db.query(Topic).filter(Topic.id == topic_id).first()
                if not topic:
                    topic = Topic(
                        id=topic_id,
                        unit_id=unit_id,
                        name=topic_name,
                        aliases=topic_name.lower(),
                        est_study_minutes=est_mins
                    )
                    db.add(topic)
                    total_topics_created += 1

    db.commit()
    print(f"[SyllabusParser] Successfully ingested {total_units_created} units and {total_topics_created} topics into database.")

if __name__ == "__main__":
    init_db()
    db_session = SessionLocal()
    ingest_curriculum_and_syllabus(db_session)
    db_session.close()

