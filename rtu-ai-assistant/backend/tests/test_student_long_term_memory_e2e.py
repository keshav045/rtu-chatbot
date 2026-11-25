"""
End-to-End Test Suite for RTU Student Long-Term Memory System.

Tests:
1. Student profile creation and academic fact storage.
2. Struggle detection from conversational queries (marking weak topics).
3. Mastery breakthrough detection (updating confidence scores).
4. System prompt memory block formatting and relevance injection.
5. Selective memory forgetting and complete reset lifecycle.
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, User, StudentMemory, TopicMastery, Branch, SessionModel
from app.database.session import SessionLocal
from app.main import app
from app.retrieval.pipeline import RetrievalPipeline
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.services.quiz_service import QuizService
from app.core.config import settings


@pytest.fixture
def memory_db():
    """Isolated in-memory SQLite database for testing student long-term memory."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSession()

    # Seed branch
    cse = Branch(id="BRANCH_CSE", code="CSE", name="Computer Science & Engineering")
    db.add(cse)
    db.commit()

    try:
        yield db
    finally:
        db.close()


def test_student_profile_and_mastery_initialization(memory_db):
    """Test user creation and baseline long-term context block generation."""
    user = MemoryService.get_or_create_user(
        db=memory_db,
        user_id="test_student_01",
        name="Rohit Sharma",
        branch_code="CSE",
        year=3,
        language_pref="hinglish"
    )
    assert user.id == "test_student_01"
    assert user.year == 3
    assert user.language_pref == "hinglish"

    ctx = MemoryService.get_student_memory_context(memory_db, user_id="test_student_01", target_subject="DBMS")
    assert "[STUDENT LONG-TERM ACADEMIC MEMORY]" in ctx["prompt_block"]
    assert "Year 3" in ctx["prompt_block"]
    assert "DBMS" in ctx["prompt_block"]


def test_struggle_detection_updates_weak_areas(memory_db):
    """Verify that expressing difficulty in a query autonomously marks the topic as weak."""
    user_id = "test_student_02"
    query = "I am struggling with B-tree split condition during insertion."
    
    MemoryService.record_turn_interaction(
        db=memory_db,
        user_id=user_id,
        query=query,
        answer="A B-tree node splits when keys reach order m.",
        subject="DSA",
        intent="academic"
    )

    ctx = MemoryService.get_student_memory_context(memory_db, user_id=user_id, target_subject="DSA")
    assert len(ctx["weak_topics"]) >= 1
    assert any("B-tree" in w or "split" in w.lower() for w in ctx["weak_topics"])
    assert "Identified Weak Areas:" in ctx["prompt_block"]


def test_mastery_signal_improves_confidence(memory_db):
    """Verify that student breakthrough or positive quiz marks improve confidence."""
    user_id = "test_student_03"
    
    # Initial practice
    MemoryService.update_topic_mastery(
        db=memory_db,
        user_id=user_id,
        subject_code="DBMS",
        topic_name="Normalization 3NF",
        confidence_delta=0.0,
        is_weak=True
    )

    # Student acknowledges understanding
    MemoryService.record_turn_interaction(
        db=memory_db,
        user_id=user_id,
        query="Got it, makes sense now! 3NF is clear.",
        answer="Excellent work! Now let's try a past exam problem.",
        subject="DBMS",
        intent="academic"
    )

    mastery = memory_db.query(TopicMastery).filter(TopicMastery.user_id == user_id).first()
    assert mastery.confidence_score > 0.35


def test_goal_extraction_and_prompt_injection(memory_db):
    """Test extracting exam dates and study milestones into persistent memory."""
    user_id = "test_student_04"
    MemoryService.record_turn_interaction(
        db=memory_db,
        user_id=user_id,
        query="My DBMS exam is tomorrow and I need to pass my backlog.",
        answer="Don't worry, let's focus on the highest-yield 15-mark questions.",
        subject="DBMS"
    )

    ctx = MemoryService.get_student_memory_context(memory_db, user_id=user_id, target_subject="DBMS")
    assert len(ctx["facts"]) >= 1
    assert "Study Goals & Milestones:" in ctx["prompt_block"]
    assert "exam is tomorrow" in ctx["prompt_block"].lower()


def test_memory_clear_lifecycle(memory_db):
    """Test full memory reset for a student."""
    user_id = "test_student_05"
    MemoryService.add_explicit_memory(memory_db, user_id, "academic_fact", "Likes tree algorithms")
    MemoryService.update_topic_mastery(memory_db, user_id, "DSA", "Binary Heaps", is_weak=True)

    res = MemoryService.clear_student_memory(memory_db, user_id)
    assert res["deleted_memories"] == 1
    assert res["deleted_topic_masteries"] == 1

    ctx = MemoryService.get_student_memory_context(memory_db, user_id=user_id)
    assert len(ctx["weak_topics"]) == 0
    assert len(ctx["facts"]) == 0


def test_quiz_evaluation_updates_topic_mastery():
    """Verify QuizService grades persist TopicMastery updates in long-term memory."""
    uid = f"quiz_mem_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        quiz_svc = QuizService()
        q = quiz_svc.get_next_question(db, "DSA")
        assert "question_text" in q

        # High-scoring answer against deterministic DSA topic -> not weak
        good_ans = (
            "AVL tree and B-Tree are self-balancing binary search tree data structures. "
            "Step 1 check balance factor or node keys, step 2 apply rotation or split. "
            "For example inserting nodes 10, 20, 30 achieves O(log n) search complexity "
            "as illustrated in the labeled tree diagram."
        )
        res = quiz_svc.evaluate_answer(
            db=db,
            question_id="unknown",
            question="Explain AVL Tree and B-Tree with rotations.",
            topic_id="TOPIC_DSA_U4_03",
            student_answer=good_ans,
            target_marks=10,
            user_id=uid,
        )
        assert res["score"] >= 6.0
        assert res.get("topic_mastery") is not None
        masteries = MemoryService.get_all_topic_masteries(db, uid)
        assert len(masteries) == 1
        assert masteries[0]["is_weak_area"] is False

        # Low-scoring answer on same topic -> flagged as weak area
        bad_res = quiz_svc.evaluate_answer(
            db=db,
            question_id="unknown",
            question="Explain AVL Tree and B-Tree with rotations.",
            topic_id="TOPIC_DSA_U4_03",
            student_answer="idk",
            target_marks=10,
            user_id=uid,
        )
        assert bad_res["score"] == 0.0
        masteries_after = MemoryService.get_all_topic_masteries(db, uid)
        assert masteries_after[0]["is_weak_area"] is True
    finally:
        MemoryService.clear_student_memory(db, uid)
        u = db.query(User).filter(User.id == uid).first()
        if u:
            db.delete(u)
            db.commit()
        db.close()


def test_rejected_out_of_syllabus_query_does_not_pollute_memory():
    """Verify TopicPaperVerifier-rejected off-topic queries are not stored as episodic memory."""
    uid = f"rej_mem_{uuid.uuid4().hex[:8]}"
    sid = f"sess_{uuid.uuid4().hex[:8]}"
    db = SessionLocal()
    try:
        pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        pipeline.load_indexes(db)
        chat_svc = ChatService(pipeline=pipeline)
        res = chat_svc.process_turn(
            db=db,
            user_message="How to make pasta at home?",
            session_id=sid,
            filters={"subject": "OS"},
            provider_name="offline",
            user_id=uid,
        )
        assert res["screening"]["is_in_syllabus"] is False
        assert MemoryService.get_all_memories(db, uid) == []
    finally:
        sess = db.query(SessionModel).filter(SessionModel.id == sid).first()
        if sess:
            db.delete(sess)
        MemoryService.clear_student_memory(db, uid)
        u = db.query(User).filter(User.id == uid).first()
        if u:
            db.delete(u)
            db.commit()
        else:
            db.commit()
        db.close()


def test_memory_rest_api_crud():
    """Verify GET/POST/DELETE /api/memory and /api/memory/mastery REST endpoints."""
    uid = f"api_mem_{uuid.uuid4().hex[:8]}"
    with TestClient(app) as client:
        post_mem = client.post("/api/memory", json={
            "user_id": uid,
            "category": "study_goal",
            "content": "Finish DBMS Unit 3 normalization before Friday",
            "importance": 1.8,
        })
        assert post_mem.status_code == 200, post_mem.text
        mem_id = post_mem.json()["memory"]["id"]

        post_mast = client.post("/api/memory/mastery", json={
            "user_id": uid,
            "subject_code": "DBMS",
            "topic_name": "BCNF Normalization",
            "confidence_delta": -0.1,
            "is_weak": True,
            "notes": "Needs decomposition practice",
        })
        assert post_mast.status_code == 200, post_mast.text

        get_res = client.get(f"/api/memory?user_id={uid}&subject=DBMS")
        assert get_res.status_code == 200, get_res.text
        body = get_res.json()
        assert body["user_id"] == uid
        assert len(body["memories"]) == 1
        assert len(body["topic_mastery"]) == 1
        assert "BCNF Normalization" in body["weak_topics"]

        del_one = client.delete(f"/api/memory/{mem_id}")
        assert del_one.status_code == 200

        del_all = client.delete(f"/api/memory?user_id={uid}")
        assert del_all.status_code == 200
        assert del_all.json()["deleted_topic_masteries"] == 1

    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == uid).first()
        if u:
            db.delete(u)
            db.commit()
    finally:
        db.close()
