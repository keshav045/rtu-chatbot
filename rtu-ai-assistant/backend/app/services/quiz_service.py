"""
Interactive Quiz & Answer Evaluation Service for RTU Exam Helper Chatbot
Implements Section 1.1, 8.1, and 10 of Implementation Plan Version 2.
Generates questions from syllabus topics and past papers, evaluates student answers,
scores them (0-10), and provides constructive senior feedback.
"""

import logging
import uuid
import re
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.models import Topic, Question, QuizAttempt, Subject, Paper, SessionModel, Unit
from app.llm.llm_factory import get_llm_provider
from app.services.memory_service import MemoryService

logger = logging.getLogger("rtu.quiz_service")

class QuizService:
    def get_next_question(
        self,
        db: Session,
        subject_id: str,
        topic_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetches or generates the next practice question for a subject/topic.
        Prioritizes real past RTU examination questions when available.
        """
        subj = db.query(Subject).filter(
            (Subject.id == subject_id) |
            (func.lower(Subject.short_name) == subject_id.lower()) |
            (Subject.code == subject_id)
        ).first()

        past_q = None
        if topic_id:
            past_q = db.query(Question).filter(Question.topic_id == topic_id).order_by(func.random()).first()

        if not past_q and subj:
            # Query authentic RTU exam question for this subject
            past_q = (
                db.query(Question)
                .join(Paper, Paper.id == Question.paper_id)
                .filter(Paper.subject_id == subj.id)
                .order_by(func.random())
                .first()
            )

        if past_q:
            paper_source = past_q.paper.source_pdf if past_q.paper else "Official RTU Paper"
            return {
                "question_id": past_q.id,
                "topic_id": past_q.topic_id,
                "topic_name": past_q.topic.name if past_q.topic else (subj.name if subj else "General Concept"),
                "question_text": past_q.text,
                "target_marks": past_q.marks,
                "part": past_q.part,
                "source": f"{paper_source} ({past_q.marks} Marks)",
                "exam_tip": "Focus on formal definition, labeled diagram, and step-by-step points."
            }

        # Fallback to topic from syllabus
        clean_subj = subject_id.replace("SUBJ_", "").upper()
        t_query = db.query(Topic)
        if topic_id:
            t_query = t_query.filter(Topic.id == topic_id)
        else:
            t_query = t_query.filter(Topic.id.like(f"TOPIC_{clean_subj}%"))
        topic = t_query.first()

        topic_name = topic.name if topic else (subj.name if subj else "Key Concept")
        return {
            "question_id": f"GEN_{uuid.uuid4().hex[:8]}",
            "topic_id": topic.id if topic else None,
            "topic_name": topic_name,
            "question_text": f"Explain the fundamental concepts and working mechanism of {topic_name}. Discuss with a diagram.",
            "target_marks": 8,
            "part": "Part B",
            "source": "Syllabus Standard Benchmark (Part B)",
            "exam_tip": "Structure your answer with: 1. Definition, 2. Diagram / Flow, 3. Practical Example."
        }

    @staticmethod
    def _resolve_quiz_context(
        db: Session,
        user_id: Optional[str],
        session_id: Optional[str],
        topic_id: Optional[str],
        question_id: Optional[str],
        question_text: str,
    ) -> tuple[str, str, str]:
        """Resolves (effective_user_id, subject_code, topic_name) for TopicMastery updates."""
        sess_obj = db.query(SessionModel).filter(SessionModel.id == session_id).first() if session_id else None
        eff_user_id = (
            (user_id.strip() if user_id and user_id.strip() else None)
            or (sess_obj.user_id if sess_obj and sess_obj.user_id else None)
            or MemoryService.DEFAULT_USER_ID
        )

        subject_code = "GENERAL"
        topic_name = ""

        if question_id and question_id != "unknown":
            q_rec = db.query(Question).filter(Question.id == question_id).first()
            if q_rec:
                if q_rec.topic and q_rec.topic.name:
                    topic_name = q_rec.topic.name
                if q_rec.paper and q_rec.paper.subject and q_rec.paper.subject.short_name:
                    subject_code = q_rec.paper.subject.short_name.upper()

        if topic_id and (not topic_name or subject_code == "GENERAL"):
            t_rec = db.query(Topic).filter(Topic.id == topic_id).first()
            if t_rec:
                if not topic_name and t_rec.name:
                    topic_name = t_rec.name
                if subject_code == "GENERAL" and t_rec.unit_id:
                    unit_rec = db.query(Unit).filter(Unit.id == t_rec.unit_id).first()
                    if unit_rec and unit_rec.subject and unit_rec.subject.short_name:
                        subject_code = unit_rec.subject.short_name.upper()

        if subject_code == "GENERAL" and sess_obj and sess_obj.subject_id:
            subj_rec = db.query(Subject).filter(Subject.id == sess_obj.subject_id).first()
            if subj_rec and subj_rec.short_name:
                subject_code = subj_rec.short_name.upper()

        if not topic_name:
            cleaned = re.sub(r"\s+", " ", (question_text or "").strip())
            if cleaned and cleaned != "RTU Examination Practice Question":
                topic_name = cleaned[:60]
            else:
                topic_name = f"{subject_code} Quiz Practice" if subject_code != "GENERAL" else "General Quiz Practice"

        return eff_user_id, subject_code, topic_name

    @classmethod
    def _record_quiz_mastery(
        cls,
        db: Session,
        user_id: Optional[str],
        session_id: Optional[str],
        topic_id: Optional[str],
        question_id: Optional[str],
        question_text: str,
        score: float,
        max_marks: float,
    ) -> Optional[Dict[str, Any]]:
        """Updates persistent TopicMastery in long-term memory from quiz performance."""
        try:
            eff_user_id, subject_code, topic_name = cls._resolve_quiz_context(
                db=db,
                user_id=user_id,
                session_id=session_id,
                topic_id=topic_id,
                question_id=question_id,
                question_text=question_text,
            )
            MemoryService.get_or_create_user(db, user_id=eff_user_id)
            score_10 = (score / max_marks) * 10.0 if max_marks > 0 else 0.0
            is_weak = score_10 < 6.0
            delta = 0.25 if score_10 >= 8.0 else (0.1 if score_10 >= 6.0 else -0.2)
            mastery = MemoryService.update_topic_mastery(
                db=db,
                user_id=eff_user_id,
                subject_code=subject_code,
                topic_name=topic_name,
                confidence_delta=delta,
                is_weak=is_weak,
                notes=f"Quiz score: {score:.1f}/{max_marks:.0f} ({score_10:.1f}/10)",
            )
            return {
                "user_id": mastery.user_id,
                "subject_code": mastery.subject_code,
                "topic_name": mastery.topic_name,
                "confidence_score": round(mastery.confidence_score, 2),
                "is_weak_area": mastery.is_weak_area,
            }
        except Exception as err:
            logger.debug(f"Quiz topic mastery sync skipped: {err}")
            return None

    def evaluate_answer(
        self,
        db: Session,
        student_answer: str,
        question_id: Optional[str] = None,
        question: Optional[str] = None,
        topic_id: Optional[str] = None,
        session_id: Optional[str] = None,
        target_marks: int = 10,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates a student's answer using RTU examination rubrics:
        1. Checks topic relevance against question keywords
        2. Detects refusal / non-answers
        3. Evaluates 4 structural pillars: Definition, Diagram, Steps/Complexity, and Example
        4. Scales score to question target marks
        """
        # Resolve question and topic if question_id is provided
        if question_id and question_id != "unknown":
            q_rec = db.query(Question).filter(Question.id == question_id).first()
            if q_rec:
                question = q_rec.text
                topic_id = q_rec.topic_id
                target_marks = q_rec.marks or target_marks or 10

        question_text = question or "RTU Examination Practice Question"
        max_marks = float(target_marks or 10)

        ans_clean = (student_answer or "").strip()
        ans_lower = ans_clean.lower()
        word_count = len(ans_clean.split())

        # 1. Empty or trivial answer check
        if not ans_clean or word_count < 4:
            mastery_info = self._record_quiz_mastery(
                db, user_id, session_id, topic_id, question_id, question_text, 0.0, max_marks
            )
            return {
                "score": 0.0,
                "max_score": max_marks,
                "feedback": "Answer is too brief or empty. In RTU examinations, examiners look for technical terminology, complete definitions, and labeled diagrams.",
                "senior_feedback": "Answer is too brief or empty. In RTU examinations, examiners look for technical terminology, complete definitions, and labeled diagrams.",
                "missing_points": ["Formal technical definition", "Diagram / state table", "Steps / Complexity"],
                "rubric_breakdown": {
                    "definition": 0.0,
                    "diagram": 0.0,
                    "working_steps": 0.0,
                    "example": 0.0
                },
                "senior_exam_tip": "In RTU papers, always attempt questions by stating the formal definition and drawing a block diagram.",
                "topic_mastery": mastery_info,
            }

        # 2. Refusal / "I don't know" check
        if re.search(r'\b(no idea|dont know|don\'t know|idk|nahi pata|just guessing|no answer|cannot answer)\b', ans_lower):
            mastery_info = self._record_quiz_mastery(
                db, user_id, session_id, topic_id, question_id, question_text, 0.0, max_marks
            )
            return {
                "score": 0.0,
                "max_score": max_marks,
                "feedback": "You indicated you don't know the answer. In RTU exams, even writing the basic definition or related formulas can secure 1-2 partial marks. Revise this topic with the Senior Tutor!",
                "senior_feedback": "You indicated you don't know the answer. In RTU exams, even writing the basic definition or related formulas can secure 1-2 partial marks. Revise this topic with the Senior Tutor!",
                "missing_points": ["Technical definition of core concept", "Neat labeled diagram / architecture", "Working steps and algorithm complexity"],
                "rubric_breakdown": {
                    "definition": 0.0,
                    "diagram": 0.0,
                    "working_steps": 0.0,
                    "example": 0.0
                },
                "senior_exam_tip": "Never leave a question sheet blank in RTU examinations. Always write the relevant formulas or draw the general architecture.",
                "topic_mastery": mastery_info,
            }

        # 3. Question Relevance Check
        q_words = set(re.findall(r'[a-zA-Z]{4,}', question_text.lower()))
        stopwords = {
            'what', 'explain', 'describe', 'discuss', 'with', 'write', 'about',
            'their', 'different', 'following', 'analyze', 'fundamental', 'concept',
            'concepts', 'brief', 'short', 'notes', 'examine', 'illustrate'
        }
        q_keywords = q_words - stopwords

        ans_words = set(re.findall(r'[a-zA-Z]{4,}', ans_lower))
        overlap = q_keywords.intersection(ans_words)

        # If question has distinct technical terms but answer shares zero overlap
        if len(q_keywords) >= 2 and len(overlap) == 0:
            mastery_info = self._record_quiz_mastery(
                db, user_id, session_id, topic_id, question_id, question_text, 0.5, max_marks
            )
            return {
                "score": 0.5,
                "max_score": max_marks,
                "feedback": "Your answer does not appear relevant to the question asked. Please address the specific topic, algorithm, or definition requested by the examiner.",
                "senior_feedback": "Your answer does not appear relevant to the question asked. Please address the specific topic, algorithm, or definition requested by the examiner.",
                "missing_points": ["Relevant technical keywords from the question", "Direct conceptual definition", "Exam-specific working steps"],
                "rubric_breakdown": {
                    "definition": 0.5,
                    "diagram": 0.0,
                    "working_steps": 0.0,
                    "example": 0.0
                },
                "senior_exam_tip": "Read the question carefully. RTU examiners quickly scan for key terms directly matching the question.",
                "topic_mastery": mastery_info,
            }

        # 4. Rubric Pillars Evaluation
        has_definition = bool(re.search(
            r'\b(is a|is an|is defined|refers to|technique|method|data structure|algorithm|protocol|model|system)\b',
            ans_lower
        ))
        has_diagram = bool(re.search(
            r'\b(diagram|figure|graph|trace|table|schematic|architecture|tree|node|state|block)\b',
            ans_lower
        ))
        has_steps = bool(re.search(
            r'\b(step|algorithm|procedure|operation|complexity|o\(|big-o|recurrence|case|phase)\b',
            ans_lower
        ))
        has_example = bool(re.search(
            r'\b(example|instance|e\.g\.|sample|let|input|for example|trace array)\b',
            ans_lower
        ))

        # Pillar 1: Definition (0 to 2.5)
        score_def = 1.0 if has_definition else 0.5
        if len(overlap) >= 2:
            score_def += 1.5
        elif len(overlap) == 1:
            score_def += 0.8
        score_def = min(2.5, score_def)

        # Pillar 2: Diagram / Schema (0 to 2.5)
        score_diag = 2.5 if has_diagram else 0.0

        # Pillar 3: Procedural / Working Steps (0 to 3.0)
        score_steps = 1.5 if has_steps else 0.5
        if word_count > 60:
            score_steps += 1.5
        elif word_count > 30:
            score_steps += 0.8
        score_steps = min(3.0, score_steps)

        # Pillar 4: Concrete Example / Derivation (0 to 2.0)
        score_ex = 2.0 if has_example else 0.0

        raw_total = score_def + score_diag + score_steps + score_ex
        # Scale to target_marks
        scaled_score = min(max_marks, round(raw_total * (max_marks / 10.0), 1))

        missing = []
        if not has_diagram:
            missing.append("Labeled block diagram / trace table (In RTU, neat diagrams hold 2.5 marks)")
        if not has_example:
            missing.append("Concrete trace example, code snippet, or worked calculation")
        if not has_steps:
            missing.append("Step-by-step algorithmic breakdown or time complexity")

        feedback_text = (
            f"Solid attempt! You scored {scaled_score:.1f}/{max_marks:.0f}. "
            f"Technical definition and core keywords are well-represented. "
        )
        if missing:
            feedback_text += f"To secure full marks in RTU, remember to add: {'; '.join(missing)}."
        else:
            feedback_text += "Great job including structured steps, example, and diagram reference!"

        # Persist in quiz_attempts if session_id corresponds to a valid Session row
        if session_id and db.query(SessionModel).filter(SessionModel.id == session_id).first():
            try:
                attempt = QuizAttempt(
                    id=f"quiz_{uuid.uuid4().hex[:12]}",
                    session_id=session_id,
                    topic_id=topic_id,
                    question=question_text,
                    answer=student_answer,
                    score=scaled_score,
                    feedback=feedback_text
                )
                db.add(attempt)
                db.commit()
            except Exception:
                db.rollback()

        mastery_info = self._record_quiz_mastery(
            db, user_id, session_id, topic_id, question_id, question_text, scaled_score, max_marks
        )

        scale_factor = max_marks / 10.0
        return {
            "score": scaled_score,
            "max_score": max_marks,
            "feedback": feedback_text,
            "senior_feedback": feedback_text,
            "missing_points": missing if missing else ["None! Complete answer structure."],
            "rubric_breakdown": {
                "definition": round(score_def * scale_factor, 1),
                "diagram": round(score_diag * scale_factor, 1),
                "working_steps": round(score_steps * scale_factor, 1),
                "example": round(score_ex * scale_factor, 1)
            },
            "senior_exam_tip": "In RTU handwritten sheets, always draw a neat box around final formulas and label all axes or graph nodes.",
            "topic_mastery": mastery_info,
        }
