"""
Neural Reranked Retrieval Evaluation Module (eval_retriever_with_reranker.py).
Evaluates full retrieval pipeline: Dense (FAISS) + Sparse (BM25) + RRF + Cross-Encoder Reranking.
Metrics: Hit Rate@1, Hit Rate@5, MRR, Top Rerank Scores, Reranker Latency.
"""

import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import hit_rate_at_k, mrr_at_k, keyword_overlap_score, latency_percentiles

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())

def is_doc_relevant(doc: Dict[str, Any], case: Dict[str, Any]) -> bool:
    if case.get("subject") and doc.get("subject_short", "").upper() != case["subject"].upper():
        return False
    keywords = case.get("expected_keywords", [])
    min_hits = case.get("min_keyword_hits", 1)
    text_norm = norm(doc.get("question_text", ""))
    hits = sum(1 for kw in keywords if kw and norm(kw) in text_norm)
    return hits >= min_hits

def eval_single_reranked(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    query = item["query"]
    subject = item.get("subject")

    retrieval_res = harness.pipeline.retrieve(query=query, subject=subject, top_k=5)
    retrieved_docs = retrieval_res.get("results", [])

    relevance_flags = [is_doc_relevant(doc, item) for doc in retrieved_docs]
    combined_text = " ".join([d.get("question_text", "") for d in retrieved_docs])
    kw_score = keyword_overlap_score(combined_text, item.get("expected_keywords", []))

    first_rel_rank = next((idx + 1 for idx, rel in enumerate(relevance_flags) if rel), None)
    mrr = 1.0 / first_rel_rank if first_rel_rank else 0.0
    top_score = retrieved_docs[0].get("rerank_score", -999.0) if retrieved_docs else -999.0

    return {
        "query": query,
        "retrieved_count": len(retrieved_docs),
        "top_1_score": round(top_score, 3),
        "keyword_coverage": kw_score,
        "hit_at_1": bool(relevance_flags and relevance_flags[0]),
        "hit_at_5": any(relevance_flags[:5]),
        "reciprocal_rank": mrr
    }

def run_reranked_eval(limit: int = 50) -> Dict[str, Any]:
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="Neural Reranked Hybrid Retrieval (FAISS + BM25 + CrossEncoder)",
        eval_fn=eval_single_reranked,
        limit=limit,
        filter_fn=lambda x: not x.get("is_out_of_scope", False)
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    kw_scores = [r.get("keyword_coverage", 0.0) for r in results]
    hits_at_1 = [1.0 if r.get("hit_at_1") else 0.0 for r in results]
    hits_at_5 = [1.0 if r.get("hit_at_5") else 0.0 for r in results]
    mrrs = [r.get("reciprocal_rank", 0.0) for r in results]
    top_scores = [r.get("top_1_score", 0.0) for r in results if r.get("top_1_score", -999.0) > -900]

    summary = {
        "benchmark": "eval_retriever_with_reranker",
        "total_queries": len(results),
        "hit_rate_at_1": round(sum(hits_at_1) / max(len(hits_at_1), 1), 3),
        "hit_rate_at_5": round(sum(hits_at_5) / max(len(hits_at_5), 1), 3),
        "mrr": round(sum(mrrs) / max(len(mrrs), 1), 3),
        "avg_keyword_coverage": round(sum(kw_scores) / max(len(kw_scores), 1), 3),
        "avg_top_rerank_score": round(sum(top_scores) / max(len(top_scores), 1), 3) if top_scores else None,
        "latency_profile": latency_percentiles(latencies)
    }

    print("\n=================== RERANKED RETRIEVAL RESULTS ===================")
    print(f"Total Academic Queries Evaluated: {summary['total_queries']}")
    print(f"Hit Rate @ 1 (Top-1 Precision):   {summary['hit_rate_at_1'] * 100:.1f}%")
    print(f"Hit Rate @ 5:                     {summary['hit_rate_at_5'] * 100:.1f}%")
    print(f"MRR (Mean Reciprocal Rank):       {summary['mrr']:.3f}")
    print(f"Average Keyword Coverage:         {summary['avg_keyword_coverage'] * 100:.1f}%")
    if summary['avg_top_rerank_score'] is not None:
        print(f"Mean Top Rerank Score:            {summary['avg_top_rerank_score']}")
    print(f"Mean Latency:                     {summary['latency_profile']['mean']} ms")
    print(f"P90 Latency:                      {summary['latency_profile']['p90']} ms")
    print("==================================================================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Neural Reranked Retrieval")
    parser.add_argument("--limit", type=int, default=50, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_reranked_eval(limit=args.limit)
