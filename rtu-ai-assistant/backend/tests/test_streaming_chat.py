"""Tests for SSE streaming chat endpoint and ChatService.stream_turn."""

import json
import uuid
from unittest.mock import patch
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from app.main import app
from app.database.session import SessionLocal
from app.database.models import Citation, Message, SessionModel, User
from app.retrieval.pipeline import RetrievalPipeline
from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService
from app.core.config import settings


def _parse_sse_events(resp) -> list:
    events = []
    for line in resp.iter_lines():
        if not line:
            continue
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        text = text.strip()
        if text.startswith("data:"):
            events.append(json.loads(text[5:].strip()))
    return events


def _cleanup_user_and_sessions(db, sids, uids=()):
    for sid in sids:
        if not sid:
            continue
        mids = [m.id for m in db.query(Message.id).filter(Message.session_id == sid).all()]
        if mids:
            db.query(Citation).filter(Citation.message_id.in_(mids)).delete(synchronize_session=False)
        db.query(Message).filter(Message.session_id == sid).delete(synchronize_session=False)
        db.query(SessionModel).filter(SessionModel.id == sid).delete(synchronize_session=False)
    for uid in uids:
        MemoryService.clear_student_memory(db, uid)
        u = db.query(User).filter(User.id == uid).first()
        if u:
            db.delete(u)
    db.commit()


def test_chat_stream_sse_academic_and_rejected():
    uid = f"stream_usr_{uuid.uuid4().hex[:8]}"
    sids = []
    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={
                "message": "Explain AVL tree rotations",
                "provider": "offline",
                "user_id": uid,
                "filters": {"subject": "DSA"},
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            events = _parse_sse_events(resp)

        types = [e.get("type") for e in events]
        assert "meta" in types
        assert "token" in types
        assert types[-1] == "done"

        tokens_joined = "".join(e.get("content", "") for e in events if e.get("type") == "token")
        done_evt = events[-1]
        sids.append(done_evt.get("session_id"))
        assert done_evt["answer"] == tokens_joined
        assert len(done_evt["citations"]) > 0
        assert done_evt["screening"]["is_in_syllabus"] is True

        # Verify persisted into SQLite conversation history
        conv_res = client.get(f"/api/conversations/{done_evt['session_id']}")
        assert conv_res.status_code == 200
        assert len(conv_res.json()["messages"]) == 2

        # Out-of-syllabus query must still stream polite refusal without memory pollution
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={
                "message": "how to bake a chocolate cake",
                "provider": "offline",
                "user_id": uid,
            },
        ) as rej_resp:
            assert rej_resp.status_code == 200
            rej_events = _parse_sse_events(rej_resp)

        rej_done = rej_events[-1]
        assert rej_done["type"] == "done"
        sids.append(rej_done.get("session_id"))
        assert rej_done["screening"]["is_in_syllabus"] is False
        assert len(rej_done["citations"]) == 0

    db = SessionLocal()
    try:
        # Only the valid DSA query should exist in episodic memory, never the cake query
        mems = MemoryService.get_all_memories(db, uid)
        assert all("cake" not in m["content"].lower() for m in mems)
        _cleanup_user_and_sessions(db, sids, [uid])
    finally:
        db.close()


def test_chat_service_stream_turn_with_langchain_model():
    db = SessionLocal()
    sid = f"stream_lc_{uuid.uuid4().hex[:8]}"
    try:
        pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        pipeline.load_indexes(db)
        chat_svc = ChatService(pipeline=pipeline)

        fake_llm = GenericFakeChatModel(
            messages=iter([AIMessage(content="Tutor Answer:\nBinary Search divides the sorted array in half each step. Time complexity is O(log n).")])
        )
        with patch("app.services.chat_service.get_llm", return_value=fake_llm), patch(
            "app.services.chat_service.resolve_auto_providers", return_value=["openai"]
        ):
            events = list(
                chat_svc.stream_turn(
                    db=db,
                    user_message="Explain Binary Search complexity",
                    session_id=sid,
                    provider_name="openai",
                    filters={"subject": "DSA"},
                )
            )

        token_chunks = [e["content"] for e in events if e.get("type") == "token"]
        done_evt = events[-1]
        assert done_evt["type"] == "done"
        assert len(token_chunks) >= 2
        assert "Tutor Answer:" not in done_evt["answer"]
        assert "O(log n)" in done_evt["answer"]
        assert "".join(token_chunks) == done_evt["answer"]
    finally:
        _cleanup_user_and_sessions(db, [sid])
        db.close()
