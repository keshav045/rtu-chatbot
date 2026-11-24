"""
Long-Term Memory & Academic Profile Service for RTU AI Academic Assistant.

Provides persistent, cross-session memory management:
1. User-Centric Academic Profile (Year, Branch, Target Subjects, Language Mode)
2. Topic Mastery & Weak Spots Tracking (Confidence scoring, attempt counters, struggle detection)
3. Episodic & Factual Memory Store (Study goals, exam deadlines, past breakthroughs)
4. Contextual Prompt Block Generator for Senior Tutor prompt injection
5. Heuristic & Semantic Extraction of student learning signals from conversational turns
"""

import re
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database.models import User, StudentMemory, TopicMastery, Branch, Subject


class MemoryService:
    """Manages student long-term memory, topic mastery, and prompt augmentation."""

    DEFAULT_USER_ID = "student_default"

    # Regex patterns for detecting student goals, exam timelines, and topic struggles
    GOAL_PATTERNS = [
        re.compile(r"\b(exam|paper)\s+is\s+(tomorrow|next week|in \d+ days|coming up)\b", re.IGNORECASE),
        re.compile(r"\b(target|aiming for|want to score|need)\s+(\d+%\+?|\d+\s*marks|pointer|sgpa)\b", re.IGNORECASE),
        re.compile(r"\b(clear|pass)\s+(my\s+)?(backlog|back|supplementary)\s*(in\s+([a-zA-Z\s]+))?", re.IGNORECASE),
        re.compile(r"\b(crash course|revision plan|study plan)\s+for\s+([a-zA-Z\s]+)", re.IGNORECASE),
    ]

    STRUGGLE_PATTERNS = [
        re.compile(r"\b(don't understand|do not understand|confused about|struggling with|having trouble with|hard to grasp)\s+([a-zA-Z0-9\s\-]+)", re.IGNORECASE),
        re.compile(r"\b(can you explain again|explain simply|not clear|still confused)\b", re.IGNORECASE),
        re.compile(r"\b(always forget|difficult topic|weak in)\s+([a-zA-Z0-9\s\-]+)", re.IGNORECASE),
    ]

    MASTERY_PATTERNS = [
        re.compile(r"\b(got it|understood|makes sense now|clear now|crystal clear|i understand)\b", re.IGNORECASE),
        re.compile(r"\b(thank you|thanks senior|great explanation)\b", re.IGNORECASE),
    ]

    @classmethod
    def get_or_create_user(
        cls,
        db: Session,
        user_id: Optional[str] = None,
        name: Optional[str] = None,
        branch_code: str = "CSE",
        year: int = 2,
        language_pref: str = "english"
    ) -> User:
        """Retrieves existing user or initializes a default student profile."""
        uid = (user_id or cls.DEFAULT_USER_ID).strip()
        user = db.query(User).filter(User.id == uid).first()
        if not user:
            # Ensure branch exists or resolve ID
            branch_obj = db.query(Branch).filter(Branch.code == branch_code.upper()).first()
            user = User(
                id=uid,
                name=name or f"RTU Student ({uid[:8]})",
                branch_id=branch_obj.id if branch_obj else None,
                year=year,
                language_pref=language_pref,
                college="Rajasthan Technical University (RTU)",
                created_at=datetime.utcnow()
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user

    @classmethod
    def get_student_memory_context(
        cls,
        db: Session,
        user_id: Optional[str] = None,
        target_subject: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Builds a structured dictionary and prompt injection block for the Senior Tutor.
        """
        uid = (user_id or cls.DEFAULT_USER_ID).strip()
        user = db.query(User).filter(User.id == uid).first()
        if not user:
            user = cls.get_or_create_user(db, user_id=uid)

        # Retrieve high-priority academic facts and study goals
        memories = (
            db.query(StudentMemory)
            .filter(StudentMemory.user_id == uid)
            .order_by(desc(StudentMemory.importance_weight), desc(StudentMemory.updated_at))
            .limit(6)
            .all()
        )

        # Retrieve topic mastery (prioritizing active weak areas)
        from sqlalchemy import func
        mastery_query = db.query(TopicMastery).filter(TopicMastery.user_id == uid)
        if target_subject:
            clean_sub = target_subject.replace("SUBJ_", "").strip().upper()
            mastery_query = mastery_query.filter(
                (func.upper(TopicMastery.subject_code) == clean_sub) |
                (TopicMastery.subject_code == "GENERAL")
            )

        weak_topics = (
            mastery_query.filter(TopicMastery.is_weak_area == True)
            .order_by(TopicMastery.confidence_score.asc())
            .limit(4)
            .all()
        )

        mastered_topics = (
            mastery_query.filter(TopicMastery.is_weak_area == False, TopicMastery.confidence_score >= 0.7)
            .order_by(TopicMastery.confidence_score.desc())
            .limit(3)
            .all()
        )

        # Construct concise prompt block
        facts_list = [m.content for m in memories if m.category in ("academic_fact", "study_goal")]
        episodic_list = [m.content for m in memories if m.category == "episodic_summary"]
        weak_names = [f"{w.topic_name} (needs review)" for w in weak_topics]
        mastered_names = [m.topic_name for m in mastered_topics]

        branch_name = user.branch.name if user.branch else (user.branch_id or "Computer Science")

        lines = ["[STUDENT LONG-TERM ACADEMIC MEMORY]"]
        lines.append(f"- Student Profile: Year {user.year} | Branch: {branch_name} | Preferred Tone: {user.language_pref.capitalize()}")
        
        if target_subject:
            lines.append(f"- Active Subject: {target_subject}")
        if weak_names:
            lines.append(f"- Identified Weak Areas: {', '.join(weak_names)}")
        if mastered_names:
            lines.append(f"- Mastered Concepts: {', '.join(mastered_names)}")
        if facts_list:
            lines.append(f"- Study Goals & Milestones: {'; '.join(facts_list[:3])}")
        if episodic_list:
            lines.append(f"- Recent Discussion Context: {'; '.join(episodic_list[:2])}")
        
        lines.append("- Coaching Directive: Patiently guide student through weak concepts when touched upon. Do not explicitly recite internal profile tags.")

        prompt_block = "\n".join(lines)

        return {
            "user_id": uid,
            "branch": branch_name,
            "year": user.year,
            "language_pref": user.language_pref,
            "weak_topics": [w.topic_name for w in weak_topics],
            "mastered_topics": [m.topic_name for m in mastered_topics],
            "facts": facts_list,
            "prompt_block": prompt_block
        }

    @classmethod
    def record_turn_interaction(
        cls,
        db: Session,
        user_id: Optional[str],
        query: str,
        answer: str,
        subject: Optional[str] = None,
        intent: Optional[str] = "academic",
        quiz_score: Optional[float] = None
    ) -> None:
        """
        Extracts learning signals and persists long-term facts & topic mastery.
        """
        if not user_id:
            user_id = cls.DEFAULT_USER_ID
        cls.get_or_create_user(db, user_id=user_id)

        clean_sub = (subject or "GENERAL").replace("SUBJ_", "").strip().upper()
        clean_q = query.strip()

        # 1. Extract study goals / exam timelines
        for pat in cls.GOAL_PATTERNS:
            match = pat.search(clean_q)
            if match:
                fact_content = f"Student noted: '{clean_q[:90]}'"
                cls._upsert_memory(db, user_id, "study_goal", fact_content, importance=1.5)
                break

        # 2. Extract topic struggle signals
        detected_struggle = False
        for pat in cls.STRUGGLE_PATTERNS:
            match = pat.search(clean_q)
            if match:
                detected_struggle = True
                extracted_topic = match.group(2).strip() if match.lastindex and match.lastindex >= 2 else ""
                topic_label = cls._clean_topic_string(extracted_topic) or clean_q[:40]
                cls.update_topic_mastery(
                    db=db,
                    user_id=user_id,
                    subject_code=clean_sub,
                    topic_name=topic_label,
                    confidence_delta=-0.25,
                    is_weak=True,
                    notes=f"Student expressed confusion: '{clean_q[:80]}'"
                )
                break

        # 3. Extract mastery / breakthrough signals
        if not detected_struggle:
            for pat in cls.MASTERY_PATTERNS:
                if pat.search(clean_q):
                    recent_topic = (
                        db.query(TopicMastery)
                        .filter(TopicMastery.user_id == user_id, TopicMastery.subject_code == clean_sub)
                        .order_by(desc(TopicMastery.last_studied_at))
                        .first()
                    )
                    target_topic_name = recent_topic.topic_name if recent_topic else clean_q[:40]
                    cls.update_topic_mastery(
                        db=db,
                        user_id=user_id,
                        subject_code=clean_sub,
                        topic_name=target_topic_name,
                        confidence_delta=0.2,
                        is_weak=False
                    )
                    break

        # 4. Handle explicit quiz score if provided
        if quiz_score is not None:
            is_weak = quiz_score < 6.0
            delta = 0.3 if quiz_score >= 8.0 else (-0.3 if quiz_score < 5.0 else 0.0)
            cls.update_topic_mastery(
                db=db,
                user_id=user_id,
                subject_code=clean_sub,
                topic_name=f"{clean_sub} Quiz Evaluation",
                confidence_delta=delta,
                is_weak=is_weak,
                notes=f"Quiz score: {quiz_score}/10"
            )

        # 5. Summarize substantial academic questions into episodic memory
        if intent in ("academic", "EXAM_QUESTION") and len(clean_q) > 20:
            summary = f"Discussed {clean_sub}: {clean_q[:70]}"
            cls._upsert_memory(db, user_id, "episodic_summary", summary, importance=0.8)

    @classmethod
    def update_topic_mastery(
        cls,
        db: Session,
        user_id: str,
        subject_code: str,
        topic_name: str,
        confidence_delta: float = 0.0,
        is_weak: Optional[bool] = None,
        notes: Optional[str] = None
    ) -> TopicMastery:
        """Creates or updates topic competency entry."""
        clean_topic = cls._clean_topic_string(topic_name) or topic_name.strip()[:100]
        if not clean_topic:
            clean_topic = "General Subject Questions"

        mastery = (
            db.query(TopicMastery)
            .filter(
                TopicMastery.user_id == user_id,
                TopicMastery.subject_code == subject_code,
                TopicMastery.topic_name.ilike(f"%{clean_topic[:30]}%")
            )
            .first()
        )

        if not mastery:
            init_conf = 0.35 if is_weak else 0.6
            mastery = TopicMastery(
                id=f"tm_{uuid.uuid4().hex[:12]}",
                user_id=user_id,
                subject_code=subject_code,
                topic_name=clean_topic,
                confidence_score=max(0.0, min(1.0, init_conf + confidence_delta)),
                attempts_count=1,
                is_weak_area=is_weak if is_weak is not None else False,
                notes=notes,
                last_studied_at=datetime.utcnow()
            )
            db.add(mastery)
        else:
            mastery.attempts_count += 1
            mastery.confidence_score = max(0.0, min(1.0, mastery.confidence_score + confidence_delta))
            if is_weak is not None:
                mastery.is_weak_area = is_weak
            if notes:
                mastery.notes = notes
            mastery.last_studied_at = datetime.utcnow()

        db.commit()
        db.refresh(mastery)
        return mastery

    @classmethod
    def _upsert_memory(
        cls,
        db: Session,
        user_id: str,
        category: str,
        content: str,
        importance: float = 1.0
    ) -> StudentMemory:
        """Inserts a memory item or updates an existing matching entry."""
        clean_c = content.strip()
        existing = (
            db.query(StudentMemory)
            .filter(
                StudentMemory.user_id == user_id,
                StudentMemory.category == category,
                StudentMemory.content == clean_c
            )
            .first()
        )
        if existing:
            existing.importance_weight = max(existing.importance_weight, importance)
            existing.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(existing)
            return existing

        new_mem = StudentMemory(
            id=f"mem_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            category=category,
            content=clean_c,
            importance_weight=importance,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        db.add(new_mem)
        db.commit()
        db.refresh(new_mem)
        return new_mem

    @classmethod
    def add_explicit_memory(
        cls,
        db: Session,
        user_id: str,
        category: str,
        content: str,
        importance: float = 1.0
    ) -> Dict[str, Any]:
        """User or API-initiated memory record creation."""
        cls.get_or_create_user(db, user_id=user_id)
        mem = cls._upsert_memory(db, user_id, category, content, importance)
        return {
            "id": mem.id,
            "category": mem.category,
            "content": mem.content,
            "importance": mem.importance_weight,
            "updated_at": mem.updated_at.isoformat()
        }

    @classmethod
    def get_all_memories(cls, db: Session, user_id: str) -> List[Dict[str, Any]]:
        """Lists all persistent memories for the user."""
        records = (
            db.query(StudentMemory)
            .filter(StudentMemory.user_id == user_id)
            .order_by(desc(StudentMemory.updated_at))
            .all()
        )
        return [
            {
                "id": r.id,
                "category": r.category,
                "content": r.content,
                "importance": r.importance_weight,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None
            }
            for r in records
        ]

    @classmethod
    def get_all_topic_masteries(cls, db: Session, user_id: str) -> List[Dict[str, Any]]:
        """Lists all topic competency tracking records for the user."""
        records = (
            db.query(TopicMastery)
            .filter(TopicMastery.user_id == user_id)
            .order_by(TopicMastery.is_weak_area.desc(), desc(TopicMastery.last_studied_at))
            .all()
        )
        return [
            {
                "id": r.id,
                "subject_code": r.subject_code,
                "topic_name": r.topic_name,
                "confidence_score": round(r.confidence_score, 2),
                "attempts_count": r.attempts_count,
                "is_weak_area": r.is_weak_area,
                "notes": r.notes,
                "last_studied_at": r.last_studied_at.isoformat() if r.last_studied_at else None
            }
            for r in records
        ]

    @classmethod
    def clear_student_memory(cls, db: Session, user_id: str) -> Dict[str, int]:
        """Clears memory and mastery records for a clean reset."""
        deleted_m = db.query(StudentMemory).filter(StudentMemory.user_id == user_id).delete()
        deleted_t = db.query(TopicMastery).filter(TopicMastery.user_id == user_id).delete()
        db.commit()
        return {"deleted_memories": deleted_m, "deleted_topic_masteries": deleted_t}

    @staticmethod
    def _clean_topic_string(text: str) -> str:
        """Normalizes noisy topic text extracted from regex matches."""
        cleaned = re.sub(r"[^\w\s\-\+]", "", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:60]
