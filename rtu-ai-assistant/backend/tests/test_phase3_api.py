"""
Pytest Suite for Phase 3: Conversational RAG, FastAPI Services & Analytics
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.main import app

@pytest.fixture(scope="module")
def client():
    # Use TestClient as context manager to trigger lifespan startup
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"]["papers"] >= 100
    assert data["database"]["questions"] >= 300
    assert data["retrieval_loaded"] is True

def test_list_subjects(client):
    response = client.get("/api/subjects")
    assert response.status_code == 200
    subjects = response.json()
    assert len(subjects) > 0
    short_codes = [s["subject_short"] for s in subjects]
    assert "DSA" in short_codes

def test_search_endpoint(client):
    payload = {
        "query": "Explain binary search tree and AVL tree rotations",
        "top_k": 3
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["results_count"] > 0
    assert len(data["results"]) <= 3
    assert "context_string" in data
    assert len(data["citations"]) > 0

def test_chat_multi_turn_flow(client):
    # Turn 1: Initial query
    turn1_payload = {
        "message": "What is AVL tree and balance factor?",
        "provider": "offline",
        "top_k": 3
    }
    res1 = client.post("/api/chat", json=turn1_payload)
    assert res1.status_code == 200
    data1 = res1.json()

    conv_id = data1["conversation_id"]
    assert conv_id is not None
    assert len(data1["assistant_message"]["content"]) > 50
    assert len(data1["citations"]) > 0

    # Turn 2: Follow-up question relying on memory
    turn2_payload = {
        "conversation_id": conv_id,
        "message": "Explain it for 10 marks",
        "provider": "offline",
        "top_k": 3
    }
    res2 = client.post("/api/chat", json=turn2_payload)
    assert res2.status_code == 200
    data2 = res2.json()

    assert data2["conversation_id"] == conv_id
    assert "AVL" in data2["rewritten_query"] or "tree" in data2["rewritten_query"].lower()
    assert "10 Marks" in data2["assistant_message"]["content"] or "Part C" in data2["assistant_message"]["content"]

    # Verify conversation retrieval
    conv_res = client.get(f"/api/conversations/{conv_id}")
    assert conv_res.status_code == 200
    conv_data = conv_res.json()
    assert len(conv_data["messages"]) == 4  # 2 user + 2 assistant

def test_papers_and_pdf_endpoints(client):
    res = client.get("/api/papers?subject=DSA")
    assert res.status_code == 200
    papers = res.json()
    assert len(papers) > 0

    first_paper = next((p for p in papers if p.get("question_count", 0) > 0), papers[0])
    paper_id = first_paper["paper_id"]

    # Detail
    detail_res = client.get(f"/api/papers/{paper_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["paper_id"] == paper_id
    assert len(detail["questions"]) > 0

    # PDF Streaming check
    pdf_res = client.get(f"/api/papers/{paper_id}/pdf")
    assert pdf_res.status_code == 200
    assert pdf_res.headers["content-type"] == "application/pdf"
    assert len(pdf_res.content) > 1000

def test_academic_analytics(client):
    # Summary
    summary_res = client.get("/api/analytics/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["total_papers"] >= 100
    assert summary["total_questions"] >= 300
    assert len(summary["semesters"]) >= 3

    # Repeated questions
    rep_res = client.get("/api/analytics/repeated?threshold=0.75&limit=10")
    assert rep_res.status_code == 200
    rep_data = rep_res.json()
    assert isinstance(rep_data, list)

