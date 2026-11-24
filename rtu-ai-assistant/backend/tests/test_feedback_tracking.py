"""
Tests for Answer Correctness & Feedback Tracking Endpoints (POST / GET / DELETE /api/feedback).
"""

import uuid
from fastapi.testclient import TestClient

from app.main import app
from app.database.session import SessionLocal
from app.database.models import Feedback, Message, SessionModel


def test_feedback_tracking_lifecycle():
    with TestClient(app) as client:
        # 1. Create a real chat turn so message_id & question/answer link in DB
        chat_res = client.post(
            "/api/chat",
            json={
                "message": "What is AVL tree balance factor?",
                "provider": "offline",
                "filters": {"subject": "DSA"},
            },
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        session_id = chat_data["session_id"]
        assistant_msg_id = chat_data["assistant_message"]["message_id"]

        # 2. Mark Incorrect ("down") initially
        fb_res1 = client.post(
            "/api/feedback",
            json={
                "session_id": session_id,
                "message_id": assistant_msg_id,
                "rating": "down",
                "comment": "Missed diagram details",
            },
        )
        assert fb_res1.status_code == 200
        fb1 = fb_res1.json()
        assert fb1["is_correct"] is False
        assert fb1["verdict"] == "Incorrect"
        feedback_id = fb1["feedback_id"]

        # 3. Flip vote to Correct ("up") -> upserts same feedback record
        fb_res2 = client.post(
            "/api/feedback",
            json={
                "session_id": session_id,
                "message_id": assistant_msg_id,
                "rating": "up",
                "comment": "Verified accurate after review",
            },
        )
        assert fb_res2.status_code == 200
        fb2 = fb_res2.json()
        assert fb2["feedback_id"] == feedback_id
        assert fb2["is_correct"] is True
        assert fb2["verdict"] == "Correct"

        # 4. GET /api/feedback returns Q&A context, subject, and accuracy summary
        list_res = client.get("/api/feedback", params={"verdict": "correct", "subject": "DSA"})
        assert list_res.status_code == 200
        payload = list_res.json()
        assert payload["summary"]["total_rated"] >= 1
        assert payload["summary"]["correct_count"] >= 1
        matched = next((r for r in payload["records"] if r["feedback_id"] == feedback_id), None)
        assert matched is not None
        assert matched["is_correct"] is True
        assert matched["verdict"] == "Correct"
        assert "AVL tree" in matched["question"]
        assert len(matched["answer"]) > 20
        assert matched["comment"] == "Verified accurate after review"

        # 5. Cleanup feedback & test session
        del_res = client.delete(f"/api/feedback/{feedback_id}")
        assert del_res.status_code == 200

        db = SessionLocal()
        try:
            sess = db.query(SessionModel).filter(SessionModel.id == session_id).first()
            if sess:
                db.delete(sess)
                db.commit()
        finally:
            db.close()
