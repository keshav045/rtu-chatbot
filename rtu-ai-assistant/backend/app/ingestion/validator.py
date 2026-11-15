"""
Metadata & Canonical Question Record Validator
Enforces strict data integrity for the RTU AI Academic Assistant.
"""

from pathlib import Path
from typing import List, Dict, Tuple
from .canonical_schema import CanonicalQuestionRecord, CanonicalPaperRecord

class MetadataValidator:
    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root
        self.seen_question_ids = set()
        self.seen_paper_ids = set()
        self.errors = []
        self.warnings = []

    def validate_paper(self, paper: CanonicalPaperRecord) -> Tuple[bool, List[str]]:
        errs = []
        if not paper.paper_id:
            errs.append("paper_id is missing.")
        if paper.paper_id in self.seen_paper_ids:
            errs.append(f"Duplicate paper_id: {paper.paper_id}")
        self.seen_paper_ids.add(paper.paper_id)

        if paper.semester not in (3, 4, 5):
            errs.append(f"Invalid semester {paper.semester} for paper {paper.paper_id}")

        if not (2012 <= paper.year <= 2026):
            errs.append(f"Invalid examination year {paper.year} for paper {paper.paper_id}")

        pdf_full_path = self.dataset_root / paper.relative_path
        if not pdf_full_path.exists():
            errs.append(f"Source PDF file does not exist: {pdf_full_path}")

        if paper.page_count <= 0:
            errs.append(f"Invalid page_count {paper.page_count}")

        if errs:
            self.errors.extend(errs)
            return False, errs
        return True, []

    def validate_question(self, question: CanonicalQuestionRecord, paper: CanonicalPaperRecord) -> Tuple[bool, List[str]]:
        errs = []
        if not question.question_id:
            errs.append("question_id is missing.")
        if question.question_id in self.seen_question_ids:
            errs.append(f"Duplicate question_id: {question.question_id}")
        self.seen_question_ids.add(question.question_id)

        if len(question.question_text.strip()) < 8:
            errs.append(f"Question text too short or empty for {question.question_id}")

        if question.marks <= 0:
            errs.append(f"Invalid marks {question.marks} for {question.question_id}")

        if not (1 <= question.page_number <= paper.page_count):
            errs.append(f"page_number {question.page_number} out of bounds (1..{paper.page_count}) for {question.question_id}")

        if errs:
            self.errors.extend(errs)
            return False, errs
        return True, []

