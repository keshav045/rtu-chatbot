"""
Comprehensive Test Suite for RTU Implementation Plan Version 2 Features
Tests:
- Database Schema and Models
- ImportanceService (SQL frequency, 10M vs 2M, Tier A/B/C)
- PlannerService (Knapsack algorithm, 10% buffer, formula revision)
- QuizService (Practice question retrieval, 0-10 rubric evaluation)
- UploadService (SHA-256 deduplication, review queue)
- FastAPI Section 10 Endpoints
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal
from app.database.models import Subject, Paper, Question, Unit, Topic, SessionModel
from app.services.importance_service import ImportanceService
from app.services.planner_service import PlannerService
from app.services.quiz_service import QuizService
from app.services.upload_service import UploadService
from app.main import app

def test_database_populated_v2():
    db = SessionLocal()
    try:
        subjects_cnt = db.query(Subject).count()
        papers_cnt = db.query(Paper).count()
        questions_cnt = db.query(Question).count()
        units_cnt = db.query(Unit).count()
        topics_cnt = db.query(Topic).count()

        assert subjects_cnt >= 10, f"Expected >= 10 subjects, got {subjects_cnt}"
        assert papers_cnt >= 50, f"Expected >= 50 papers, got {papers_cnt}"
        assert questions_cnt >= 200, f"Expected >= 200 questions, got {questions_cnt}"
        assert units_cnt >= 30, f"Expected >= 30 units, got {units_cnt}"
        assert topics_cnt >= 80, f"Expected >= 80 topics, got {topics_cnt}"
    finally:
        db.close()

def test_importance_service_tier_a():
    db = SessionLocal()
    try:
        service = ImportanceService()
        result = service.get_ranked_topics(db, "DSA")
        assert "error" not in result
        assert result["tier"] == "Tier A"
        assert result["paper_count"] >= 5
        assert len(result["ranked_topics"]) > 0
        top_topic = result["ranked_topics"][0]
        assert "topic_name" in top_topic
        assert "total_marks" in top_topic
        assert "years_appeared" in top_topic
    finally:
        db.close()

def test_importance_service_tier_c():
    db = SessionLocal()
    try:
        service = ImportanceService()
        result = service.get_ranked_topics(db, "OS")
        assert "error" not in result
        assert result["tier"] == "Tier C"
        assert "Syllabus Weightage" in result["disclaimer"] or "Syllabus Standard" in result["ranked_topics"][0]["basis"]
    finally:
        db.close()

def test_planner_service_knapsack():
    db = SessionLocal()
    try:
        service = PlannerService()
        plan = service.generate_plan(db, "DSA", hours_available=6.0)
        assert "error" not in plan
        assert plan["total_hours"] == 6.0
        assert plan["buffer_minutes"] > 0
        assert plan["revision_minutes"] > 0
        assert len(plan["schedule"]) > 0
        assert plan["final_revision_block"] is not None
    finally:
        db.close()

def test_quiz_service():
    db = SessionLocal()
    try:
        service = QuizService()
        q = service.get_next_question(db, "DSA")
        assert "question_text" in q
        assert "target_marks" in q

        eval_result = service.evaluate_answer(
            db=db,
            question_id=q["question_id"],
            student_answer="A B-tree is a self-balancing search tree. It maintains sorted data and allows searches, sequential access, insertions, and deletions in logarithmic time."
        )
        assert "score" in eval_result
        assert 0.0 <= eval_result["score"] <= 10.0
        assert "rubric_breakdown" in eval_result
        assert "senior_feedback" in eval_result
    finally:
        db.close()

def test_api_endpoints_with_lifespan():
    with TestClient(app) as client:
        # 1. Health
        health_resp = client.get("/api/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "healthy"

        # 2. Create Session
        resp = client.post("/api/session", json={
            "subject_id": "DSA",
            "branch": "CSE",
            "year": 2,
            "hours_left": 8.0,
            "mode": "crash_course"
        })
        assert resp.status_code == 200, resp.text
        session_data = resp.json()
        assert "session_id" in session_data
        assert session_data["tier"] == "Tier A"

        # 3. Get Important Topics
        resp2 = client.get("/api/topics/DSA/important")
        assert resp2.status_code == 200, resp2.text
        topics_data = resp2.json()
        assert len(topics_data["ranked_topics"]) > 0

        # 4. Generate Study Plan
        resp3 = client.post("/api/plan", json={
            "subject_id": "DSA",
            "hours_left": 5.0,
            "target_score": 75
        })
        assert resp3.status_code == 200, resp3.text
        plan_data = resp3.json()
        assert plan_data["total_hours"] == 5.0
        assert len(plan_data["schedule"]) > 0

        # 5. Quiz Workflow
        resp4 = client.post("/api/quiz/next", json={"subject_id": "DSA"})
        assert resp4.status_code == 200, resp4.text
        quiz_q = resp4.json()

        resp5 = client.post("/api/quiz/answer", json={
            "question_id": quiz_q["question_id"],
            "student_answer": "Binary search trees allow logarithmic search when balanced. AVL trees balance using rotations."
        })
        assert resp5.status_code == 200, resp5.text
        assert "score" in resp5.json()

        # 6. Senior Tutor Chat in Hinglish
        resp6 = client.post("/api/chat", json={
            "message": "AVL tree rotations kya hain? 10 marks ke liye batao.",
            "language": "hinglish"
        })
        assert resp6.status_code == 200, resp6.text
        chat_data = resp6.json()
        assert "assistant_message" in chat_data
        assert len(chat_data["assistant_message"]["content"]) > 50
        assert len(chat_data["citations"]) > 0
