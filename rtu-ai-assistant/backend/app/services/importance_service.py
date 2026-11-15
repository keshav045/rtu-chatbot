"""
Importance Ranking Service for RTU Exam Helper Chatbot
Implements Section 2, 5, and 7 of Implementation Plan Version 2.
Computes data-driven topic frequencies from past RTU papers:
- Years appeared
- Long answer count (Marks >= 8)
- Short answer count (Marks <= 2)
- Total marks
"""

from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, case

from app.database.models import Subject, Unit, Topic, Question, Paper

class ImportanceService:
    def get_ranked_topics(
        self,
        db: Session,
        subject_id: str,
        target_marks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Returns ranked topics for a subject based on past RTU examination frequency.
        Handles Tier A, Tier B, and Tier C subjects honestly.
        """
        # Resolve subject
        subj = db.query(Subject).filter(
            (Subject.id == subject_id) | 
            (func.lower(Subject.short_name) == subject_id.lower()) |
            (Subject.code == subject_id)
        ).first()

        if not subj:
            return {"error": f"Subject '{subject_id}' not found."}

        # Check subject tier
        tier = subj.data_tier or "Tier C"

        # If Tier C (Syllabus only, 0 papers):
        if tier == "Tier C":
            units = db.query(Unit).filter(Unit.subject_id == subj.id).order_by(Unit.unit_no).all()
            topics_list = []
            rank = 1
            for u in units:
                for t in u.topics:
                    topics_list.append({
                        "rank": rank,
                        "topic_id": t.id,
                        "topic_name": t.name,
                        "unit_no": u.unit_no,
                        "years_appeared": 0,
                        "long_answer_count": 0,
                        "short_answer_count": 0,
                        "total_marks": 0,
                        "est_study_minutes": t.est_study_minutes,
                        "importance_level": "Syllabus Standard",
                        "basis": "Syllabus Weightage (Tier C: 0 approved past papers)"
                    })
                    rank += 1

            return {
                "subject": subj.name,
                "subject_code": subj.code,
                "short_name": subj.short_name,
                "tier": "Tier C",
                "tier_note": "Syllabus only. Explanations, quizzes, and study plans available. No empirical importance ranking yet.",
                "disclaimer": "Syllabus Weightage mode. No past papers yet. Rankings based on syllabus standard hours.",
                "paper_count": 0,
                "total_topics": len(topics_list),
                "ranked_topics": topics_list
            }

        # Query topic frequencies for Tier A & B subjects
        # SQL aggregation:
        # SELECT t.id, t.name, u.unit_no, t.est_study_minutes,
        #   COUNT(DISTINCT p.year) as years_appeared,
        #   SUM(CASE WHEN q.marks >= 8 THEN 1 ELSE 0 END) as long_answer_count,
        #   SUM(CASE WHEN q.marks <= 2 THEN 1 ELSE 0 END) as short_answer_count,
        #   SUM(q.marks) as total_marks
        query = (
            db.query(
                Topic.id.label("topic_id"),
                Topic.name.label("topic_name"),
                Unit.unit_no.label("unit_no"),
                Topic.est_study_minutes.label("est_study_minutes"),
                func.count(func.distinct(Paper.year)).label("years_appeared"),
                func.sum(case((Question.marks >= 8, 1), else_=0)).label("long_answer_count"),
                func.sum(case((Question.marks <= 2, 1), else_=0)).label("short_answer_count"),
                func.sum(Question.marks).label("total_marks")
            )
            .join(Unit, Unit.id == Topic.unit_id)
            .join(Question, Question.topic_id == Topic.id)
            .join(Paper, Paper.id == Question.paper_id)
            .filter(Unit.subject_id == subj.id, Question.status == "approved")
            .group_by(Topic.id, Topic.name, Unit.unit_no, Topic.est_study_minutes)
        )

        if target_marks and target_marks >= 8:
            query = query.order_by(desc("long_answer_count"), desc("total_marks"), desc("years_appeared"))
        elif target_marks and target_marks <= 2:
            query = query.order_by(desc("short_answer_count"), desc("total_marks"), desc("years_appeared"))
        else:
            query = query.order_by(desc("total_marks"), desc("years_appeared"), desc("long_answer_count"))

        rows = query.all()

        ranked_topics = []
        for idx, r in enumerate(rows, 1):
            total_marks_val = int(r.total_marks or 0)
            years_val = int(r.years_appeared or 0)
            long_cnt = int(r.long_answer_count or 0)
            short_cnt = int(r.short_answer_count or 0)

            # Assign importance classification (honest multi-year threshold - D-08)
            if years_val >= 2 and total_marks_val >= 20:
                imp = "HIGH PRIORITY (Frequently Asked)"
            elif total_marks_val >= 25 or years_val >= 2 or (years_val >= 1 and total_marks_val >= 15):
                imp = "HIGH WEIGHTAGE (Core Exam Topic)"
            elif total_marks_val >= 10:
                imp = "MEDIUM PRIORITY"
            else:
                imp = "STANDARD PRIORITY"

            ranked_topics.append({
                "rank": idx,
                "topic_id": r.topic_id,
                "topic_name": r.topic_name,
                "unit_no": r.unit_no,
                "years_appeared": years_val,
                "long_answer_count": long_cnt,
                "short_answer_count": short_cnt,
                "total_marks": total_marks_val,
                "est_study_minutes": r.est_study_minutes or 45,
                "importance_level": imp,
                "basis": f"Data-backed RTU frequency across {years_val} distinct examination years"
            })

        tier_note = "Full features and data-driven topic ranking enabled." if tier == "Tier A" else "Limited data (1-4 years), treat frequencies as tentative."
        paper_cnt = len(subj.papers)

        return {
            "subject": subj.name,
            "subject_code": subj.code,
            "short_name": subj.short_name,
            "tier": tier,
            "tier_note": tier_note,
            "disclaimer": tier_note,
            "paper_count": paper_cnt,
            "total_topics": len(ranked_topics),
            "ranked_topics": ranked_topics
        }

    def get_10_vs_2_mark_guide(self, db: Session, subject_id: str) -> Dict[str, Any]:
        """Provides side-by-side categorization of topics into Long Answer vs Short Concept candidates."""
        long_ranked = self.get_ranked_topics(db, subject_id, target_marks=10)
        short_ranked = self.get_ranked_topics(db, subject_id, target_marks=2)

        return {
            "subject": long_ranked.get("subject"),
            "tier": long_ranked.get("tier"),
            "long_answer_topics": [
                t for t in long_ranked.get("ranked_topics", []) if t["long_answer_count"] > 0 or t["years_appeared"] >= 2
            ][:10],
            "short_concept_topics": [
                t for t in short_ranked.get("ranked_topics", []) if t["short_answer_count"] > 0
            ][:10]
        }
