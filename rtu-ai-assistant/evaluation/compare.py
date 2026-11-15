"""
A/B Model and Retrieval Comparison Module (compare.py).
Evaluates and outputs side-by-side comparative deltas between:
1. Baseline Retrieval (FAISS + BM25 + RRF) vs Neural Reranked Retrieval (+ Cross-Encoder)
2. Cost & Latency trade-offs across inference providers (Groq, Gemini, Offline)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from eval_retriever import run_retrieval_eval
from eval_retriever_with_reranker import run_reranked_eval
from eval_cost import run_cost_eval

def run_retrieval_comparison(limit: int = 30) -> Dict[str, Any]:
    print(f"\n[compare.py] Comparing Baseline Hybrid vs Neural Reranked Retrieval (Limit: {limit})...")
    
    baseline = run_retrieval_eval(limit=limit)
    reranked = run_reranked_eval(limit=limit)

    delta_hit_1 = round(reranked["hit_rate_at_1"] - baseline["hit_rate_at_1"], 3)
    delta_hit_5 = round(reranked["hit_rate_at_5"] - baseline["hit_rate_at_5"], 3)
    delta_mrr = round(reranked["mrr"] - baseline["mrr"], 3)
    delta_latency = round(reranked["latency_profile"]["mean"] - baseline["latency_profile"]["mean"], 1)

    comparison = {
        "benchmark": "compare_retrieval",
        "sample_size": limit,
        "baseline": {
            "hit_at_1": baseline["hit_rate_at_1"],
            "hit_at_5": baseline["hit_rate_at_5"],
            "mrr": baseline["mrr"],
            "mean_latency_ms": baseline["latency_profile"]["mean"]
        },
        "reranked": {
            "hit_at_1": reranked["hit_rate_at_1"],
            "hit_at_5": reranked["hit_rate_at_5"],
            "mrr": reranked["mrr"],
            "mean_latency_ms": reranked["latency_profile"]["mean"]
        },
        "deltas": {
            "delta_hit_at_1": delta_hit_1,
            "delta_hit_at_5": delta_hit_5,
            "delta_mrr": delta_mrr,
            "delta_latency_ms": delta_latency
        }
    }

    print("\n=================== RETRIEVAL COMPARISON MATRIX ===================")
    print(f"{'Metric':<25} | {'Baseline (RRF)':<16} | {'Neural Reranked':<16} | {'Delta':<10}")
    print("-" * 75)
    print(f"{'Hit Rate @ 1':<25} | {baseline['hit_rate_at_1']*100:6.1f}%          | {reranked['hit_rate_at_1']*100:6.1f}%          | {delta_hit_1*100:+5.1f}%")
    print(f"{'Hit Rate @ 5':<25} | {baseline['hit_rate_at_5']*100:6.1f}%          | {reranked['hit_rate_at_5']*100:6.1f}%          | {delta_hit_5*100:+5.1f}%")
    print(f"{'MRR (Reciprocal Rank)':<25} | {baseline['mrr']:6.3f}           | {reranked['mrr']:6.3f}           | {delta_mrr:+5.3f}")
    print(f"{'Mean Latency (ms)':<25} | {baseline['latency_profile']['mean']:6.1f} ms        | {reranked['latency_profile']['mean']:6.1f} ms        | {delta_latency:+6.1f} ms")
    print("===================================================================\n")

    return comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Baseline vs Neural Reranked Retrieval")
    parser.add_argument("--limit", type=int, default=30, help="Number of queries to compare")
    args = parser.parse_args()
    run_retrieval_comparison(limit=args.limit)
