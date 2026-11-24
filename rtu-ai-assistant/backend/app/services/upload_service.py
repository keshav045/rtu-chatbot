"""
Student Question Paper Upload & Processing Pipeline
Implements Section 6 of Implementation Plan Version 2.
Handles PDF validation, SHA-256 deduplication, header extraction, question splitting,
automatic topic tagging, and review queue management.
"""

import hashlib
import logging
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
import fitz

from app.database.models import Upload, Paper, Question, Subject, Topic
from app.core.config import settings
from app.retrieval.document_preparer import prepare_retrieval_documents

logger = logging.getLogger("rtu.upload")

UPLOADS_DIR = settings.STORAGE_DIR / "student_uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class UploadService:
    def __init__(self, pipeline: Optional[Any] = None):
        self.pipeline = pipeline

    def _extract_questions_from_pdf(self, pdf_path: Path) -> Tuple[int, str, List[Tuple[str, int]]]:
        """Opens PDF, returns (total_pages, first_page_text, list of (question_text, page_no))."""
        doc = fitz.open(pdf_path)
        try:
            first_page_text = doc[0].get_text() if len(doc) > 0 else ""
            total_pages = len(doc)
            extracted_questions: List[Tuple[str, int]] = []
            q_regex = re.compile(r"(?:Q\.?\s*\d+|^\d+\s*[\.\)]|\([a-z]\))\s*(.*)", re.I | re.MULTILINE)
            for pno in range(total_pages):
                p_text = doc[pno].get_text()
                for m in q_regex.findall(p_text):
                    clean_q = m.strip()
                    if len(clean_q) > 15 and "ersahilkagyan" not in clean_q.lower():
                        extracted_questions.append((clean_q, pno + 1))
            return total_pages, first_page_text, extracted_questions
        finally:
            doc.close()

    def _persist_paper_and_questions(
        self,
        db: Session,
        upload_record: Upload,
        subj_obj: Subject,
        extracted_questions: List[Tuple[str, int]],
        total_pages: int,
        file_size_kb: float,
    ) -> List[Question]:
        """Creates Paper and Question rows for an approved upload."""
        new_paper_id = f"UP_PAPER_{upload_record.id}"
        existing_paper = db.query(Paper).filter(Paper.id == new_paper_id).first()
        if existing_paper:
            return []

        saved_p = Path(upload_record.saved_path) if upload_record.saved_path else None
        rel_p = (
            str(saved_p.relative_to(settings.WORKSPACE_ROOT))
            if (saved_p and saved_p.exists() and saved_p.is_relative_to(settings.WORKSPACE_ROOT))
            else (upload_record.filename or "uploaded_paper.pdf")
        )
        new_paper = Paper(
            id=new_paper_id,
            subject_id=subj_obj.id,
            scheme_id="SCHEME_RTU_2018",
            year=upload_record.year or 2024,
            session=upload_record.session or "Main",
            source="upload",
            source_pdf=upload_record.filename,
            relative_path=rel_p,
            page_count=max(1, total_pages),
            file_size_kb=file_size_kb,
            file_hash=upload_record.file_hash,
        )
        db.add(new_paper)
        db.flush()

        fallback_topic = db.query(Topic).join(Topic.unit).filter(Topic.unit.has(subject_id=subj_obj.id)).first()
        inserted: List[Question] = []
        subj_short = subj_obj.short_name or "GEN"
        yr = upload_record.year or 2024

        for q_idx, (q_text, p_num) in enumerate(extracted_questions[:15], 1):
            part = "Part A" if len(q_text) < 80 else ("Part B" if len(q_text) < 180 else "Part C")
            marks = 2 if part == "Part A" else (8 if part == "Part B" else 15)
            q_entry = Question(
                id=f"RTU_UP_{subj_short}_{yr}_Q{q_idx:02d}_{upload_record.id}",
                paper_id=new_paper_id,
                topic_id=fallback_topic.id if fallback_topic else None,
                text=q_text,
                marks=marks,
                part=part,
                tag_confidence=0.80,
                status="approved",
                page_number=p_num,
            )
            db.add(q_entry)
            inserted.append(q_entry)
        return inserted

    def _sync_retrieval_indexes(self, db: Session) -> None:
        """BUG-14 FIX: Keep RetrievalPipeline cache/indexes and LC retriever synced after upload approval."""
        try:
            import app.retrieval.lc_retriever as lc_ret_mod
            lc_ret_mod._CACHED_RETRIEVER = None
        except Exception:
            pass

        if self.pipeline is None:
            return
        try:
            self.pipeline.clear_cache()
            all_docs = prepare_retrieval_documents(db)
            self.pipeline.doc_store = {doc["question_id"]: doc for doc in all_docs}
            # Rebuild BM25 quickly (<50ms) and append any newly approved doc vectors to FAISS
            self.pipeline.bm25_indexer.build_index(all_docs)
            dense = self.pipeline.dense_indexer
            if dense and dense.index is not None and dense.model is not None:
                existing_ids = set(dense.id_mapping)
                missing_docs = [d for d in all_docs if d["question_id"] not in existing_ids]
                if missing_docs:
                    import numpy as np
                    vecs = dense.model.encode(
                        [d["document_text"] for d in missing_docs],
                        batch_size=32,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                    ).astype(np.float32)
                    dense.index.add(vecs)
                    dense.id_mapping.extend(d["question_id"] for d in missing_docs)
        except Exception as e:
            logger.warning(f"Non-fatal upload index sync warning: {e}")

    def process_pdf_upload(
        self,
        db: Session,
        filename: str,
        file_bytes: bytes,
        subject_id: Optional[str] = None,
        year: Optional[int] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validates, saves, extracts, and tags a student uploaded question paper PDF.
        """
        # 1. SHA-256 Hash Deduplication
        sha256 = hashlib.sha256(file_bytes).hexdigest()
        existing = db.query(Upload).filter(Upload.file_hash == sha256).first()
        if existing:
            return {
                "upload_id": existing.id,
                "status": "duplicate",
                "message": "This question paper PDF has already been uploaded and processed.",
                "details": {
                    "filename": existing.filename,
                    "status": existing.status,
                    "extracted_questions": existing.extracted_questions_count
                }
            }

        # Save to disk
        upload_id = f"up_{uuid.uuid4().hex[:12]}"
        safe_filename = f"{upload_id}_{filename}"
        saved_path = UPLOADS_DIR / safe_filename
        with open(saved_path, "wb") as f:
            f.write(file_bytes)

        # 2. Extract Text & Detect Header
        total_pages, first_page_text, extracted_questions = self._extract_questions_from_pdf(saved_path)

        # Detect Subject Code & Year if not explicitly passed
        detected_year = year
        if not detected_year:
            year_match = re.search(r"\b(20[12]\d)\b", first_page_text)
            detected_year = int(year_match.group(1)) if year_match else 2024

        detected_subj_short = subject_id
        if not detected_subj_short:
            for s in ["DSA", "DBMS", "OS", "TOC", "DE", "AEM", "OOP", "SE", "DMS", "TC", "MEFA", "MI", "CN"]:
                if re.search(rf"\b{s}\b", first_page_text, re.I):
                    detected_subj_short = s
                    break

        detected_subj_short = detected_subj_short or "GEN"

        # Decision: Auto-approve or Review Queue
        status = "approved" if len(extracted_questions) >= 5 else "review_queue"
        confidence = 0.88 if status == "approved" else 0.55

        # 4. Save Upload Record
        subj_obj = db.query(Subject).filter(
            (Subject.short_name == detected_subj_short) | (Subject.id == f"SUBJ_{detected_subj_short}")
        ).first()

        # Ensure status is review_queue if subject could not be resolved
        if not subj_obj:
            status = "review_queue"

        upload_record = Upload(
            id=upload_id,
            user_id=user_id or "student_web",
            file_hash=sha256,
            status=status,
            subject_id=subj_obj.id if subj_obj else None,
            year=detected_year,
            session="Main",
            confidence=confidence,
            filename=filename,
            saved_path=str(saved_path),
            extracted_questions_count=len(extracted_questions)
        )
        db.add(upload_record)
        db.flush()

        # If auto-approved, integrate as Paper and Questions
        if status == "approved" and subj_obj:
            self._persist_paper_and_questions(
                db=db,
                upload_record=upload_record,
                subj_obj=subj_obj,
                extracted_questions=extracted_questions,
                total_pages=total_pages,
                file_size_kb=round(len(file_bytes) / 1024.0, 1),
            )

        db.commit()
        if status == "approved" and subj_obj:
            self._sync_retrieval_indexes(db)

        status_msg = (
            f"Successfully parsed {len(extracted_questions)} questions from {filename}. Status: {status.upper()} (Crowdsourced data added to RTU corpus)."
            if status == "approved"
            else f"Parsed {len(extracted_questions)} questions from {filename}. Sent to admin review queue for verification."
        )

        return {
            "upload_id": upload_id,
            "filename": filename,
            "status": status,
            "subject": detected_subj_short,
            "year": detected_year,
            "pages": total_pages,
            "questions_extracted": len(extracted_questions),
            "message": status_msg
        }

    def get_upload_status(self, db: Session, upload_id: str) -> Optional[Dict[str, Any]]:
        up = db.query(Upload).filter(Upload.id == upload_id).first()
        if not up:
            return {"error": f"Upload with id {upload_id} not found"}
        return {
            "upload_id": up.id,
            "filename": up.filename,
            "status": up.status,
            "year": up.year,
            "confidence": up.confidence,
            "questions_extracted": up.extracted_questions_count,
            "created_at": up.created_at.isoformat() if up.created_at else None
        }

    def approve_or_reject_upload(self, db: Session, upload_id: str, action: str) -> Dict[str, Any]:
        """Approves or rejects a crowdsourced upload from the review queue."""
        up = db.query(Upload).filter(Upload.id == upload_id).first()
        if not up:
            return {"error": f"Upload with id {upload_id} not found."}

        action_clean = action.strip().lower()
        if action_clean not in ["approve", "reject"]:
            return {"error": "Invalid action. Must be 'approve' or 'reject'."}

        if action_clean == "reject":
            up.status = "rejected"
            db.commit()
            return {"status": "success", "upload_id": upload_id, "action": "rejected", "message": "Upload rejected."}

        # Approve action: extract questions from saved PDF and persist Paper + Questions
        up.status = "approved"
        subj_obj = db.query(Subject).filter(Subject.id == up.subject_id).first() if up.subject_id else None
        if subj_obj:
            saved_p = Path(up.saved_path) if up.saved_path else None
            total_pages = 1
            extracted_questions: List[Tuple[str, int]] = []
            file_size_kb = 50.0
            if saved_p and saved_p.exists():
                file_size_kb = round(saved_p.stat().st_size / 1024.0, 1)
                total_pages, _, extracted_questions = self._extract_questions_from_pdf(saved_p)
                if not up.extracted_questions_count:
                    up.extracted_questions_count = len(extracted_questions)
            self._persist_paper_and_questions(
                db=db,
                upload_record=up,
                subj_obj=subj_obj,
                extracted_questions=extracted_questions,
                total_pages=total_pages,
                file_size_kb=file_size_kb,
            )

        db.commit()
        if subj_obj:
            self._sync_retrieval_indexes(db)
        return {
            "status": "success",
            "upload_id": upload_id,
            "action": "approved",
            "message": "Upload approved and integrated into RTU corpus."
        }

    def list_review_queue(self, db: Session) -> List[Dict[str, Any]]:
        uploads = db.query(Upload).filter(Upload.status == "review_queue").all()
        return [
            {
                "upload_id": u.id,
                "filename": u.filename,
                "year": u.year,
                "questions_extracted": u.extracted_questions_count,
                "created_at": u.created_at.isoformat() if u.created_at else None
            }
            for u in uploads
        ]
