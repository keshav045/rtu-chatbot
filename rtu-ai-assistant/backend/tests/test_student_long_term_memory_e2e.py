"""
End-to-End Test Suite for RTU Student Long-Term Memory System.

Tests:
1. Student profile creation and academic fact storage.
2. Struggle detection from conversational queries (marking weak topics).
3. Mastery breakthrough detection (updating confidence scores).
4. System prompt memory block formatting and relevance injection.
5. Selective memory forgetting and complete reset lifecycle.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base, User, StudentMemory, TopicMastery, Branch
from app.services.memory_service import MemoryService


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
