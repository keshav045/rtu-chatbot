"""
Pytest Test Suite for Phase 2: Hybrid Retrieval & Reranker Engine
Tests FAISS dense search, BM25 keyword search, RRF fusion, CrossEncoder reranking, and metadata filtering.
"""

import pytest
from pathlib import Path
from app.database.session import SessionLocal
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.hybrid_retriever import reciprocal_rank_fusion, apply_metadata_filters

@pytest.fixture(scope="module")
def pipeline():
    db = SessionLocal()
    pipe = RetrievalPipeline()
    pipe.load_indexes(db)
    yield pipe
    db.close()

def test_faiss_dense_search(pipeline):
    query = "Quick sort time complexity and partitioning"
    results = pipeline.dense_indexer.search(query, top_k=5)
    assert len(results) > 0
    top_qid, score = results[0]
    assert top_qid in pipeline.doc_store
    assert score > 0.0

def test_bm25_exact_code_and_keyword_search(pipeline):
    # Test subject code search
    code_results = pipeline.bm25_indexer.search("3E1202", top_k=5)
    assert len(code_results) > 0
    # Top result should belong to Data Structures & Algorithms
    top_doc = pipeline.doc_store[code_results[0][0]]
    assert top_doc["subject_code"] in ("3E1202", "3CS4-05")
    assert top_doc["subject_short"] == "DSA"

    # Test exact keyword search
    sort_results = pipeline.bm25_indexer.search("bubble sort", top_k=5)
    assert len(sort_results) > 0

def test_rrf_hybrid_fusion():
    dense_ranking = [("docA", 0.9), ("docB", 0.8), ("docC", 0.7)]
    sparse_ranking = [("docB", 12.0), ("docD", 8.0), ("docA", 4.0)]

    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking], k=60)
    assert len(fused) == 4
    # docB was rank 2 in dense and rank 1 in sparse -> should have highest fused score
    top_doc_id, top_score = fused[0]
    assert top_doc_id == "docB"

def test_metadata_filtering(pipeline):
    # Filter by subject DSA only
    res_dsa = pipeline.retrieve("Explain algorithm", subject="DSA", top_k=5)
    for doc in res_dsa["results"]:
        assert doc["subject_short"] == "DSA"

    # Filter by semester 3 only
    res_sem3 = pipeline.retrieve("Explain algorithm", semester=3, top_k=5)
    for doc in res_sem3["results"]:
        assert doc["semester"] == 3

    # Filter by section Part B
    res_part_b = pipeline.retrieve("Explain algorithm", section="Part B", top_k=5)
    for doc in res_part_b["results"]:
        assert "part b" in doc["section"].lower()

def test_cross_encoder_reranking_accuracy(pipeline):
    query = "Explain bubble sort and quick sort algorithm with complexity"
    res = pipeline.retrieve(query, subject="DSA", top_k=3)
    assert res["results_count"] > 0
    top_hit = res["results"][0]
    assert "quick sort" in top_hit["question_text"].lower() or "sort" in top_hit["question_text"].lower()
    assert top_hit["rerank_score"] > 0.0

def test_citations_and_context_builder(pipeline):
    query = "Write Chebyshev's inequality"
    res = pipeline.retrieve(query, top_k=2)
    assert len(res["citations"]) > 0
    cit = res["citations"][0]
    assert "source_pdf" in cit
    assert cit["source_pdf"].endswith(".pdf")
    assert cit["page_number"] >= 1
    assert "context_string" in res
    assert "[Source 1" in res["context_string"]

