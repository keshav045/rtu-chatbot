"""
Academic Intelligence & Analytics Service for RTU AI Academic Assistant
Provides academic insights: repeated questions across years, year-wise comparisons,
subject distributions, and question patterns according to Plan v2 schema.
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
import numpy as np

from app.database.models import Paper, Question, Subject, Topic, SubjectOffering
from app.retrieval.dense_indexer import DenseIndexer

class AnalyticsService:
    def __init__(self, dense_indexer: Optional[DenseIndexer] = None):
        self.dense_indexer = dense_indexer

    def get_summary(self, db: Session) -> Dict[str, Any]:
        """Returns overall statistical summary of the indexed RTU dataset."""
        total_papers = db.query(Paper).count()
        total_questions = db.query(Question).count()
        total_subjects = db.query(Subject).count()

        # Semester breakdown via SubjectOffering
        sem_counts = (
            db.query(SubjectOffering.semester, func.count(SubjectOffering.id))
            .group_by(SubjectOffering.semester)
            .order_by(SubjectOffering.semester)
            .all()
        )

        # Section / Part breakdown
        sec_counts = (
            db.query(Question.part, func.count(Question.id))
            .group_by(Question.part)
            .order_by(Question.part)
            .all()
        )

        # Subject breakdown
        subj_rows = db.query(Subject).all()
        subj_data = []
        for s in subj_rows:
            q_cnt = db.query(Question).join(Paper, Paper.id == Question.paper_id).filter(Paper.subject_id == s.id).count()
            p_cnt = len(s.papers)
            subj_data.append({
                "code": s.code,
                "short_name": s.short_name,
                "name": s.name,
                "data_tier": s.data_tier,
                "paper_count": p_cnt,
                "question_count": q_cnt
            })
        subj_data.sort(key=lambda x: x["question_count"], reverse=True)

        # Year distribution
        year_counts = (
            db.query(Paper.year, func.count(Question.id))
            .join(Question, Question.paper_id == Paper.id)
            .group_by(Paper.year)
            .order_by(Paper.year)
            .all()
        )

        return {
            "total_papers": total_papers,
            "total_questions": total_questions,
            "total_subjects": total_subjects,
            "semesters": {f"Semester {sem}": count for sem, count in sem_counts},
            "sections": {sec: count for sec, count in sec_counts if sec},
            "years": {year: count for year, count in year_counts if year},
            "subjects": subj_data
        }

    def get_subject_analysis(self, db: Session, subject_short: str) -> Dict[str, Any]:
        """Provides in-depth analysis for a specific subject."""
        subj = db.query(Subject).filter(
            (func.lower(Subject.short_name) == subject_short.lower()) |
            (Subject.code == subject_short) |
            (Subject.id == subject_short)
        ).first()

        if not subj:
            return {"error": f"Subject '{subject_short}' not found in database."}

        questions = (
            db.query(Question)
            .join(Paper, Paper.id == Question.paper_id)
            .filter(Paper.subject_id == subj.id)
            .all()
        )

        if not questions:
            return {
                "subject": subj.name,
                "subject_short": subj.short_name,
                "subject_code": subj.code,
                "data_tier": subj.data_tier,
                "total_questions": 0,
                "years_covered": [],
                "marks_distribution": {},
                "section_distribution": {},
                "sample_questions": []
            }

        years = sorted(list({q.paper.year for q in questions if q.paper and q.paper.year}))
        marks_dist = {}
        section_dist = {}
        for q in questions:
            marks_dist[q.marks] = marks_dist.get(q.marks, 0) + 1
            sec = q.part or "Part A"
            section_dist[sec] = section_dist.get(sec, 0) + 1

        return {
            "subject": subj.name,
            "subject_short": subj.short_name,
            "subject_code": subj.code,
            "data_tier": subj.data_tier,
            "total_questions": len(questions),
            "years_covered": years,
            "marks_distribution": marks_dist,
            "section_distribution": section_dist,
            "sample_questions": [
                {
                    "question_id": q.id,
                    "question_text": q.text,
                    "year": q.paper.year if q.paper else None,
                    "section": q.part,
                    "marks": q.marks,
                    "source_pdf": q.paper.source_pdf if q.paper else ""
                }
                for q in questions[:10]
            ]
        }

    def find_repeated_questions(
        self,
        db: Session,
        subject_short: Optional[str] = None,
        threshold: float = 0.80,
        limit: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Identifies repeated or highly similar questions across different examination years
        using dense semantic embeddings (cosine similarity).
        """
        if not self.dense_indexer or self.dense_indexer.index is None:
            return []

        query = db.query(Question).join(Paper, Paper.id == Question.paper_id)
        if subject_short:
            query = query.join(Subject, Subject.id == Paper.subject_id).filter(
                func.lower(Subject.short_name) == subject_short.lower()
            )
        questions = query.all()

        q_map = {q.id: q for q in questions}
        id_mapping = self.dense_indexer.id_mapping or []
        id_set = set(id_mapping)
        rev_id_map = {qid: idx for idx, qid in enumerate(id_mapping)}

        q_ids = [q.id for q in questions if q.id in id_set]

        if len(q_ids) < 2:
            return []

        indices = [rev_id_map[qid] for qid in q_ids if qid in rev_id_map]

        try:
            vectors = np.array([self.dense_indexer.index.reconstruct(i) for i in indices])
            sim_matrix = np.dot(vectors, vectors.T)
        except Exception:
            return []

        results = []
        seen_pairs = set()

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    qid_a = q_ids[i]
                    qid_b = q_ids[j]
                    qa = q_map[qid_a]
                    qb = q_map[qid_b]

                    year_a = qa.paper.year if qa.paper else None
                    year_b = qb.paper.year if qb.paper else None
                    subj_name = qa.paper.subject.short_name if qa.paper and qa.paper.subject else "RTU"

                    pair_key = tuple(sorted([qid_a, qid_b]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        results.append({
                            "similarity": round(sim, 4),
                            "subject": subj_name,
                            "question_a": {
                                "question_id": qa.id,
                                "text": qa.text,
                                "year": year_a,
                                "section": qa.part,
                                "marks": qa.marks,
                                "source_pdf": qa.paper.source_pdf if qa.paper else ""
                            },
                            "question_b": {
                                "question_id": qb.id,
                                "text": qb.text,
                                "year": year_b,
                                "section": qb.part,
                                "marks": qb.marks,
                                "source_pdf": qb.paper.source_pdf if qb.paper else ""
                            },
                            "cross_year": year_a != year_b
                        })

        results.sort(key=lambda x: (x["cross_year"], x["similarity"]), reverse=True)
        return results[:limit]

    def compare_years(self, db: Session, subject_short: str, year_a: int, year_b: int) -> Dict[str, Any]:
        """Compares examination questions for a subject between two different years."""
        qa = (
            db.query(Question)
            .join(Paper, Paper.id == Question.paper_id)
            .join(Subject, Subject.id == Paper.subject_id)
            .filter(func.lower(Subject.short_name) == subject_short.lower(), Paper.year == year_a)
            .all()
        )
        qb = (
            db.query(Question)
            .join(Paper, Paper.id == Question.paper_id)
            .join(Subject, Subject.id == Paper.subject_id)
            .filter(func.lower(Subject.short_name) == subject_short.lower(), Paper.year == year_b)
            .all()
        )

        return {
            "subject": subject_short,
            "year_a": {
                "year": year_a,
                "total_questions": len(qa),
                "questions": [
                    {"question_id": q.id, "section": q.part, "marks": q.marks, "text": q.text}
                    for q in qa
                ]
            },
            "year_b": {
                "year": year_b,
                "total_questions": len(qb),
                "questions": [
                    {"question_id": q.id, "section": q.part, "marks": q.marks, "text": q.text}
                    for q in qb
                ]
            }
        }
