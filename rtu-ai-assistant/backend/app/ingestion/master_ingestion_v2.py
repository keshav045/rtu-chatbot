"""
Master Curriculum & Question Ingestion Pipeline (Version 2)
Implements Sections 5, 6, and 7 of the RTU Exam Helper Chatbot Implementation Plan.
- Seeds Universities, Schemes, Branches, Subjects, Offerings, Units, Topics
- Ingests all 123 Papers and Extracted Questions
- Computes Subject Data Tiers (Tier A, Tier B, Tier C)
- Tags Questions to Syllabus Topics
- Builds Topic Frequency Statistics
"""

import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
CLEAN_DATASET_DIR = PROJECT_ROOT / "cleaned_dataset"
CATALOG_PATH = CLEAN_DATASET_DIR / "metadata" / "dataset_catalog.csv"
EXTRACTED_TEXT_DIR = CLEAN_DATASET_DIR / "extracted_text"
SYLLABUS_PDF = PROJECT_ROOT / "syllabus" / "Syllabus_2018onwards.pdf"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal, reset_db
from app.database.models import (
    University, Scheme, Branch, Subject, SubjectOffering, Unit, Topic, Paper, Question, Chunk
)
from app.ingestion.syllabus_parser import (
    seed_foundational_metadata, parse_syllabus_pdf, extract_topics_from_unit_text, CORE_SUBJECT_MAP
)

def run_master_ingestion():
    print("=" * 70)
    print("RTU EXAM HELPER CHATBOT: MASTER CURRICULUM & DATASET INGESTION (V2)")
    print("=" * 70)

    # 1. Reset and Recreate SQLite Database
    reset_db()
    db = SessionLocal()

    # 2. Seed University, Schemes, Branches
    seed_foundational_metadata(db)

    # 3. Register Subjects & Offerings
    print("\n[1/5] Registering Core Subjects & Branch Offerings...")
    for short_code, info in CORE_SUBJECT_MAP.items():
        subj_id = f"SUBJ_{short_code}"
        subj = Subject(
            id=subj_id,
            code=info["code"],
            name=info["name"],
            short_name=short_code,
            data_tier="Tier C"  # Default, elevated later
        )
        db.add(subj)
        db.flush()

        for bcode in ["CSE", "IT"]:
            offering = SubjectOffering(
                id=f"OFFER_{short_code}_{bcode}",
                subject_id=subj_id,
                branch_id=f"BRANCH_{bcode}",
                semester=info["sem"],
                scheme_id="SCHEME_RTU_2018"
            )
            db.add(offering)

    db.commit()
    print(f"  Registered {len(CORE_SUBJECT_MAP)} core subjects across CSE and IT branches.")

    # 4. Ingest Syllabus Units & Topics
    print("\n[2/5] Parsing RTU Syllabus PDF & Extracting Units and Topics...")
    extracted_syllabus = parse_syllabus_pdf(SYLLABUS_PDF)
    print(f"  Extracted syllabus contents for {len(extracted_syllabus)} course modules.")

    total_units = 0
    total_topics = 0
    subject_topics_map = defaultdict(list)

    for item in extracted_syllabus:
        name = item["name"]
        code = item["code"]
        
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
                total_units += 1

            topic_items = extract_topics_from_unit_text(unit_content)
            for t_idx, (t_name, est_mins) in enumerate(topic_items, 1):
                topic_id = f"TOPIC_{matched_short}_U{unit_no}_{t_idx:02d}"
                topic = db.query(Topic).filter(Topic.id == topic_id).first()
                if not topic:
                    topic = Topic(
                        id=topic_id,
                        unit_id=unit_id,
                        name=t_name,
                        aliases=t_name.lower(),
                        est_study_minutes=est_mins
                    )
                    db.add(topic)
                    db.flush()
                    total_topics += 1
                    subject_topics_map[matched_short].append((topic_id, t_name))

    db.commit()
    print(f"  Ingested {total_units} Units and {total_topics} Topics from Syllabus.")

    # 5. Ingest 123 Past Examination Papers
    print("\n[3/5] Loading 123 Past RTU Examination Papers from Catalog...")
    catalog_df = pd.read_csv(CATALOG_PATH)
    papers_stored = 0

    for _, row in catalog_df.iterrows():
        short_code = str(row["subject_short"]).upper()
        subj_id = f"SUBJ_{short_code}"
        if not db.query(Subject).filter(Subject.id == subj_id).first():
            # If subject not in map, create fallback
            subj = Subject(
                id=subj_id,
                code=str(row["subject_code"]),
                name=str(row["subject_name"]),
                short_name=short_code,
                data_tier="Tier C"
            )
            db.add(subj)
            db.flush()

        paper = Paper(
            id=str(row["paper_id"]),
            subject_id=subj_id,
            scheme_id="SCHEME_RTU_2018",
            year=int(row["year"]),
            session=str(row.get("exam_session", "Main")),
            source="admin",
            source_pdf=str(row["clean_filename"]),
            relative_path=str(row["clean_relative_path"]),
            page_count=int(row["page_count"]),
            file_size_kb=float(row["file_size_kb"]),
            file_hash=str(row["md5_hash"])
        )
        db.add(paper)
        papers_stored += 1

    db.commit()
    print(f"  Stored {papers_stored} official RTU examination papers in SQLite.")

    # 6. Ingest & Tag Questions to Topics
    print("\n[4/5] Ingesting & Tagging Past Examination Questions to Syllabus Topics...")
    json_files = list(EXTRACTED_TEXT_DIR.glob("*.json"))
    print(f"  Found {len(json_files)} papers with OCR question text.")

    questions_stored = 0
    from sentence_transformers import SentenceTransformer
    tagger_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # Pre-embed topics for each subject to allow fast topic tagging
    topic_embeddings_cache = {}
    for short_code, t_list in subject_topics_map.items():
        if t_list:
            t_names = [t[1] for t in t_list]
            t_vecs = tagger_model.encode(t_names, normalize_embeddings=True)
            topic_embeddings_cache[short_code] = (t_list, t_vecs)

    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            paper_meta = data.get("metadata", {})
            paper_id = paper_meta.get("paper_id")
            paper_obj = db.query(Paper).filter(Paper.id == paper_id).first()
            if not paper_obj:
                continue

            short_code = paper_obj.subject.short_name
            questions = data.get("questions", [])
            pages_raw = data.get("pages_raw", [])

            # Get topics for this subject
            cached = topic_embeddings_cache.get(short_code)

            for q_idx, q in enumerate(questions, 1):
                raw_text = q.get("question_text", "").strip()
                if len(raw_text) < 8 or "ersahilkagyan" in raw_text.lower():
                    continue

                part_label = q.get("part", "Part A")
                sec_code = "PA" if "part a" in part_label.lower() else ("PB" if "part b" in part_label.lower() else "PC")
                q_id = f"RTU_{short_code}_{paper_obj.year}_{sec_code}_Q{q_idx:02d}_{paper_obj.id}"

                # Resolve page number
                page_num = 1
                q_snippet = raw_text[:25]
                for p_raw in pages_raw:
                    for line in p_raw.get("lines", []):
                        if q_snippet in line.get("text", "") or line.get("text", "") in raw_text:
                            page_num = p_raw.get("page", 1)
                            break
                    if page_num > 1:
                        break

                # Automatic Topic Tagging via semantic embedding similarity
                assigned_topic_id = None
                confidence = 0.85
                if cached:
                    t_list, t_vecs = cached
                    q_vec = tagger_model.encode([raw_text], normalize_embeddings=True)[0]
                    sims = np.dot(t_vecs, q_vec)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= 0.40:
                        assigned_topic_id = t_list[best_idx][0]
                        confidence = round(best_sim, 3)

                q_record = Question(
                    id=q_id,
                    paper_id=paper_obj.id,
                    topic_id=assigned_topic_id,
                    text=raw_text,
                    marks=int(q.get("marks", 2)),
                    part=part_label,
                    tag_confidence=confidence,
                    status="approved",
                    page_number=page_num
                )
                db.add(q_record)
                questions_stored += 1

        except Exception as e:
            print(f"  Error processing {jf.name}: {e}")

    db.commit()
    print(f"  Ingested & tagged {questions_stored} examination questions.")

    # 7. Compute Subject Data Tiers (Section 7)
    print("\n[5/5] Computing Subject Data Tiers (Tier A / Tier B / Tier C)...")
    subjects = db.query(Subject).all()
    tier_counts = {"Tier A": 0, "Tier B": 0, "Tier C": 0}

    for s in subjects:
        # Count distinct approved exam years of papers for this subject
        distinct_years = (
            db.query(Paper.year)
            .filter(Paper.subject_id == s.id)
            .distinct()
            .count()
        )

        if distinct_years >= 5:
            tier = "Tier A"
        elif distinct_years >= 1:
            tier = "Tier B"
        else:
            tier = "Tier C"

        s.data_tier = tier
        tier_counts[tier] += 1
        print(f"  • {s.short_name:5s} ({s.code:7s}): {distinct_years:2d} paper years -> {tier}")

    db.commit()
    print(f"\nFinal Subject Tiers Summary: {tier_counts['Tier A']} Tier A, {tier_counts['Tier B']} Tier B, {tier_counts['Tier C']} Tier C")
    print("=" * 70)
    print("MASTER INGESTION COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    db.close()

if __name__ == "__main__":
    run_master_ingestion()

