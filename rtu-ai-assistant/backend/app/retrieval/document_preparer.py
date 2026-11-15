"""
Document Preparation for Retrieval
Extracts canonical question records from SQLite (Section 5 Schema) and prepares
individual searchable retrieval units with structured metadata headers.
"""

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..database.models import Question, Paper, Subject, Topic, Unit

def prepare_retrieval_documents(db: Session) -> List[Dict[str, Any]]:
    """
    Fetches all canonical questions from SQLite and formats each into a
    rich, searchable document string according to the RTU AI blueprint.
    Compatible with Plan v2 relational models.
    """
    questions = db.query(Question).filter(Question.status == "approved").all()
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
