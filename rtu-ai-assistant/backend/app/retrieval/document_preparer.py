"""
Document Preparation for Retrieval
Extracts canonical question records from SQLite (Section 5 Schema) and prepares
individual searchable retrieval units with structured metadata headers.
"""

import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..database.models import Question, Paper, Subject, Topic, Unit

_OCR_NOISE_RE = re.compile(
    r'\[\s*contd[^\]]*|contd\.{2,}|\bpage\s*\d+\s*(?:of\s*\d*)?\b|'
    r'\bvp/\d+\b|\bpaper\s*code\b|\[?\d[a-z]{1,3}\d{3,4}\]?|\[\s*\d{3,5}\s*\]',
    re.I
)
_BOILER_FRAGMENT_RE = re.compile(
    r'^[\)\(\.\:\-\s]*(?:attempt\s+any\s+\w+\s+questions|write\s*short\s*notes?\s*on|'
    r'answer\s+the\s+following|unit[\s\-]*(?:[ivx\d]+)?|details\.)[\.\:\-\s\d\(\)or]*$',
    re.I
)


def is_meaningful_question(text: str) -> bool:
    """Filters out OCR garbage fragments and empty headers while keeping short valid Part A questions."""
    if not text or not text.strip():
        return False
    cleaned = _OCR_NOISE_RE.sub(' ', text).strip()
    if _BOILER_FRAGMENT_RE.match(cleaned):
        return False
    without_markers = re.sub(
        r'\([a-z0-9\.]+\)|\[[a-z0-9\.]+\]|\b[a-z]\)|\bor\b|\bunit[\s\-]*[ivx\d]+\b',
        ' ',
        cleaned,
        flags=re.I
    ).strip()
    alpha_words = re.findall(r'[a-zA-Z]{2,}', without_markers)
    total_alpha = sum(len(w) for w in alpha_words)
    return (len(alpha_words) >= 2 and total_alpha >= 8) or (len(alpha_words) == 1 and total_alpha >= 16)


def prepare_retrieval_documents(db: Session) -> List[Dict[str, Any]]:
    """
    Fetches all canonical questions from SQLite and formats each into a
    rich, searchable document string according to the RTU AI blueprint.
    Compatible with Plan v2 relational models.
    """
    # BUG-10 FIX: Filter out OCR garbage fragments/headers while preserving valid short questions.
    questions = [
        q for q in db.query(Question).filter(Question.status == "approved", Question.text.isnot(None)).all()
        if is_meaningful_question(q.text)
    ]
    documents = []

    for q in questions:
        # Resolve relations safely
        paper = q.paper
        subject = paper.subject if paper else None
        topic = q.topic
        unit = topic.unit if topic else None

        subj_name = subject.name if subject else "Engineering Subject"
        subj_code = subject.code if subject else "RTU-SUBJ"
        subj_short = subject.short_name if subject else "SUBJ"
        year = paper.year if paper else 2022
        source_pdf = paper.source_pdf if paper else "RTU_Exam.pdf"
        relative_path = paper.relative_path if paper else ""
        semester = 3  # default or offerings
        if subject and subject.offerings:
            semester = subject.offerings[0].semester

        topic_name = topic.name if topic else "General Engineering Concept"
        unit_no = unit.unit_no if unit else 1
        part_name = q.part or "Part A"
        marks_val = q.marks or 2
        q_text = q.text or ""

        # Searchable document text with explicit metadata headers
        doc_text = (
            f"University: Rajasthan Technical University (RTU) | "
            f"Subject: {subj_name} ({subj_code}, {subj_short}) | "
            f"Year: {year} | "
            f"Semester: {semester} | "
            f"Unit: Unit {unit_no} ({topic_name}) | "
            f"Section: {part_name} | "
            f"Marks: {marks_val} | "
            f"Page: {q.page_number} | "
            f"Question Text: {q_text}"
        )

        documents.append({
            "question_id": q.id,
            "paper_id": q.paper_id,
            "university": "Rajasthan Technical University",
            "subject": subj_name,
            "subject_id": subject.id if subject else "",
            "subject_code": subj_code,
            "subject_short": subj_short,
            "branch": "CSE/IT",
            "semester": semester,
            "year": year,
            "section": part_name,
            "unit": f"Unit {unit_no}",
            "topic_id": q.topic_id,
            "topic_name": topic_name,
            "question_number": q.id.split("_")[-1] if "_" in q.id else "1",
            "marks": marks_val,
            "question_text": q_text,
            "source_pdf": source_pdf,
            "relative_path": relative_path,
            "page_number": q.page_number or 1,
            "document_text": doc_text
        })

    return documents
