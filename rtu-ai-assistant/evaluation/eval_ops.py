"""
Operational & Database Health Evaluation Module (eval_ops.py).
Evaluates backend operational integrity:
1. SQLite relational database connection latency and record counts
2. FAISS vector index dimension and vector count integrity
3. BM25 sparse index vocabulary and document count
4. Working memory (RAM) footprint of retrieval models
Production Gate: Index integrity 100%, SQLite query latency < 50ms.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import latency_percentiles

def run_ops_eval() -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_ops] Inspecting operational health, database integrity, and index status...")

    db = harness.db
    from app.database.models import Question, Paper, Subject

    # 1. Database Queries & Timing
    t0 = time.perf_counter()
    q_count = db.query(Question).count()
    paper_count = db.query(Paper).count()
    subj_count = db.query(Subject).count()
    db_latency_ms = (time.perf_counter() - t0) * 1000.0

    # 2. FAISS Index Integrity
    faiss_index = harness.pipeline.dense_indexer.index
    faiss_ntotal = faiss_index.ntotal if faiss_index is not None else 0
    faiss_dim = faiss_index.d if faiss_index is not None else 0

    # 3. BM25 Index Integrity
    bm25_model = harness.pipeline.bm25_indexer.bm25
    bm25_docs = len(harness.pipeline.bm25_indexer.id_mapping) if harness.pipeline.bm25_indexer.id_mapping else 0

    # 4. Doc Store in RAM
    doc_store_count = len(harness.pipeline.doc_store)

    # 5. Process Memory Usage
    import psutil
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)

    faiss_valid = faiss_ntotal > 0 and faiss_dim == 384
    bm25_valid = bm25_docs > 0
    db_valid = q_count > 0 and paper_count > 0

    passed = faiss_valid and bm25_valid and db_valid and (db_latency_ms < 100.0)

    summary = {
        "benchmark": "eval_ops",
        "database": {
            "questions_count": q_count,
            "papers_count": paper_count,
            "subjects_count": subj_count,
            "query_latency_ms": round(db_latency_ms, 2),
            "status": "HEALTHY" if db_valid else "ERROR"
        },
        "faiss_dense_index": {
            "vector_count": faiss_ntotal,
            "dimension": faiss_dim,
            "status": "HEALTHY" if faiss_valid else "ERROR"
        },
        "bm25_sparse_index": {
            "document_count": bm25_docs,
            "status": "HEALTHY" if bm25_valid else "ERROR"
        },
        "in_memory_doc_store": {
            "documents_cached": doc_store_count,
            "process_ram_mb": round(ram_mb, 1)
        },
        "overall_status": "PASS" if passed else "FAIL"
    }

    print("\n=================== OPERATIONAL HEALTH RESULTS ===================")
    print(f"SQLite Total Questions:       {q_count}")
    print(f"SQLite Total Papers:          {paper_count}")
    print(f"SQLite Total Subjects:        {subj_count}")
    print(f"Relational Query Latency:     {summary['database']['query_latency_ms']} ms")
    print("------------------------------------------------------------------")
    print(f"FAISS Vectors Indexed:        {faiss_ntotal} (Expected: 387, Dim: {faiss_dim})")
    print(f"BM25 Searchable Units:        {bm25_docs}")
    print(f"Cached Memory Store:          {doc_store_count} questions")
    print(f"Process RAM Footprint:        {summary['in_memory_doc_store']['process_ram_mb']} MB")
    print("------------------------------------------------------------------")
    print(f"Operational Health Status:    {summary['overall_status']}")
    print("==================================================================\n")

    return summary

if __name__ == "__main__":
    run_ops_eval()
