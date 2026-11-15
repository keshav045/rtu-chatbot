"""
System Latency and Subsystem Profiling Evaluation Module (eval_latency.py).
Benchmarks latency distribution across each stage:
1. FAISS Dense Semantic Search
2. BM25 Sparse Keyword Search
3. Reciprocal Rank Fusion (RRF)
4. Cross-Encoder Reranker
5. Context Builder & Citation Generation
6. End-to-End Chat Turn
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import latency_percentiles

BENCHMARK_PROFILING_QUERIES = [
    "Explain Dijkstra algorithm and its time complexity with adjacency list",
    "What is the difference between B-Tree and B+ Tree in DBMS?",
    "Derive the recurrence relation for Merge Sort and solve using Master Theorem",
    "Explain working principle of Transformer and define regulation and efficiency",
    "State and prove pumping lemma for regular languages in TOC",
    "Explain paging and segmentation with TLB architecture in Operating Systems",
    "Describe OSI reference model layers with protocol data units",
    "Explain virtual functions and dynamic polymorphism in C++"
]

def profile_retrieval_subsystems(harness: EvaluationHarness) -> Dict[str, Any]:
    pipe = harness.pipeline
    dense_times = []
    sparse_times = []
    rrf_times = []
    rerank_times = []
    context_times = []
    e2e_times = []

    from app.retrieval.hybrid_retriever import reciprocal_rank_fusion

    # Warmup
    pipe.retrieve("Warmup query", top_k=2)

    for q in BENCHMARK_PROFILING_QUERIES:
        t_start = time.perf_counter()

        # 1. Dense search
        t0 = time.perf_counter()
        dense_res = pipe.dense_indexer.search(q, top_k=50)
        dense_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Sparse search
        t0 = time.perf_counter()
        sparse_res = pipe.bm25_indexer.search(q, top_k=50)
        sparse_times.append((time.perf_counter() - t0) * 1000.0)

        # 3. RRF Fusion
        t0 = time.perf_counter()
        fused = reciprocal_rank_fusion([dense_res, sparse_res], k=60)
        rrf_times.append((time.perf_counter() - t0) * 1000.0)

        # 4. Cross-Encoder
        cand_docs = [pipe.doc_store[qid] for qid, _ in fused[:30] if qid in pipe.doc_store]
        t0 = time.perf_counter()
        reranked = pipe.reranker.rerank(q, cand_docs, top_k=5)
        rerank_times.append((time.perf_counter() - t0) * 1000.0)

        # 5. Context Builder
        t0 = time.perf_counter()
        _ = pipe.context_builder.build_context(reranked)
        _ = pipe.context_builder.extract_citations(reranked)
        context_times.append((time.perf_counter() - t0) * 1000.0)

        e2e_times.append((time.perf_counter() - t_start) * 1000.0)

    return {
        "faiss_dense_search": latency_percentiles(dense_times),
        "bm25_sparse_search": latency_percentiles(sparse_times),
        "rrf_fusion": latency_percentiles(rrf_times),
        "cross_encoder_rerank": latency_percentiles(rerank_times),
        "context_citation_builder": latency_percentiles(context_times),
        "e2e_retrieval_pipeline": latency_percentiles(e2e_times)
    }

def run_latency_eval() -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_latency] Profiling latency distributions across all pipeline components...")
    breakdown = profile_retrieval_subsystems(harness)

    summary = {
        "benchmark": "eval_latency",
        "sample_queries": len(BENCHMARK_PROFILING_QUERIES),
        "subsystem_latencies": breakdown,
        "p90_e2e_retrieval_ms": breakdown["e2e_retrieval_pipeline"]["p90"],
        "mean_e2e_retrieval_ms": breakdown["e2e_retrieval_pipeline"]["mean"],
        "status": "PASS" if breakdown["e2e_retrieval_pipeline"]["p90"] <= 350.0 else "WARN"
    }

    print("\n=================== LATENCY PROFILING RESULTS ===================")
    print(f"FAISS Dense Search:     Mean {breakdown['faiss_dense_search']['mean']} ms | P90 {breakdown['faiss_dense_search']['p90']} ms")
    print(f"BM25 Keyword Search:    Mean {breakdown['bm25_sparse_search']['mean']} ms | P90 {breakdown['bm25_sparse_search']['p90']} ms")
    print(f"RRF Hybrid Fusion:      Mean {breakdown['rrf_fusion']['mean']} ms | P90 {breakdown['rrf_fusion']['p90']} ms")
    print(f"Cross-Encoder Rerank:   Mean {breakdown['cross_encoder_rerank']['mean']} ms | P90 {breakdown['cross_encoder_rerank']['p90']} ms")
    print(f"Context & Citations:    Mean {breakdown['context_citation_builder']['mean']} ms | P90 {breakdown['context_citation_builder']['p90']} ms")
    print("-----------------------------------------------------------------")
    print(f"E2E Retrieval Pipeline: Mean {summary['mean_e2e_retrieval_ms']} ms | P90 {summary['p90_e2e_retrieval_ms']} ms")
    print(f"Latency Gate Status:    {summary['status']}")
    print("=================================================================\n")

    return summary

if __name__ == "__main__":
    run_latency_eval()
