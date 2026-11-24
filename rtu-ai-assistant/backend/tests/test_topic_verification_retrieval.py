"""
Tests for RTU Syllabus Topic & Past-Paper Query Verification and Related-Topic Retrieval.
Ensures:
1. Out-of-the-box queries are rejected deterministically even when an unindexed subject (e.g. OS, PHY) is active.
2. Verified RTU queries return ALL related syllabus topics across matching units/subjects.
3. Unindexed syllabus topics (e.g. OS deadlock, PHY Wave Optics) are verified and answered cleanly with topic context.
4. Indexed RTU queries (e.g. DBMS 3NF, DSA B-tree) retrieve both related topics and verified past-paper citations.
"""

from app.database.session import SessionLocal
from app.database.models import Citation, Message, SessionModel
from app.retrieval.pipeline import RetrievalPipeline
from app.services.chat_service import ChatService
from app.core.config import settings


def _cleanup_sessions(db, session_ids):
    for sid in session_ids:
        if not sid:
            continue
        mids = [m.id for m in db.query(Message.id).filter(Message.session_id == sid).all()]
        if mids:
            db.query(Citation).filter(Citation.message_id.in_(mids)).delete(synchronize_session=False)
        db.query(Message).filter(Message.session_id == sid).delete(synchronize_session=False)
        db.query(SessionModel).filter(SessionModel.id == sid).delete(synchronize_session=False)
    db.commit()


def test_topic_verifier_blocks_out_of_box_on_unindexed_subjects():
    db = SessionLocal()
    created_sids = []
    try:
        pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        pipeline.load_indexes(db)
        chat_svc = ChatService(pipeline=pipeline)

        for msg, subj in [
            ("How to make pasta at home?", "OS"),
            ("Who won the NBA championship in 2024?", None),
            ("Best tourist attractions in Jaipur for weekend trip.", None),
            ("How to build a Next.js e-commerce store with Tailwind CSS?", None),
        ]:
            res = chat_svc.process_turn(
                db=db,
                user_message=msg,
                filters={"subject": subj} if subj else {},
                provider_name="offline",
            )
            created_sids.append(res.get("session_id"))
            assert res["screening"]["is_in_syllabus"] is False, f"Expected out-of-syllabus rejection for '{msg}' ({subj})"
            assert len(res["citations"]) == 0
            assert len(res.get("related_topics", [])) == 0
    finally:
        _cleanup_sessions(db, created_sids)
        db.close()


def test_topic_verifier_retrieves_all_related_topics_on_rtu_queries():
    db = SessionLocal()
    created_sids = []
    try:
        pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        pipeline.load_indexes(db)
        chat_svc = ChatService(pipeline=pipeline)

        # Unindexed subject OS should still match RTU syllabus topics cleanly
        os_res = chat_svc.process_turn(
            db=db,
            user_message="Explain the four conditions for deadlock in Operating Systems.",
            filters={"subject": "OS"},
            provider_name="offline",
        )
        created_sids.append(os_res.get("session_id"))
        assert os_res["screening"]["is_in_syllabus"] is True
        os_topics = os_res.get("related_topics", [])
        assert len(os_topics) >= 1
        assert any("deadlock" in t["topic_name"].lower() for t in os_topics)

        # Multi-topic indexed query should return multiple related topics + past paper citations
        btree_res = chat_svc.process_turn(
            db=db,
            user_message="What is B-tree insertion? Explain the steps.",
            provider_name="offline",
        )
        created_sids.append(btree_res.get("session_id"))
        assert btree_res["screening"]["is_in_syllabus"] is True
        assert len(btree_res.get("related_topics", [])) >= 2
        assert len(btree_res.get("citations", [])) > 0
    finally:
        _cleanup_sessions(db, created_sids)
        db.close()
