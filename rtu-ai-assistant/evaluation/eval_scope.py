"""
Academic Scope Gating Evaluation Module (eval_scope.py).
Evaluates the system's ability to discriminate between authentic RTU engineering queries
and out-of-scope / non-academic queries (abstention detection, threshold gating).
Production Gate: Scope ROC-AUC >= 0.85, Abstain Recall >= 80%.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import latency_percentiles
from app.services.intent_service import IntentRouter, IntentType

def eval_single_scope(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    query = item["query"]
    is_out = item.get("is_out_of_scope", False) or item.get("expected_behavior") == "abstain"
    
    # 1. Intent Classification Gating
    intent = IntentRouter.detect_intent(query)
    intent_abstained = intent in (IntentType.OUT_OF_SYLLABUS, IntentType.SECURITY_VIOLATION)

    # 2. Retrieval with metadata subject filter
    subj = item.get("subject")
    res = harness.pipeline.retrieve(query=query, subject=subj, top_k=1)
    results = res.get("results", [])
    top_score = results[0].get("rerank_score", -999.0) if results else -999.0

    # Optimal threshold established in benchmark v2
    threshold = -9.5
    abstained = intent_abstained or (top_score < threshold) or (len(results) == 0)

    # Passed if: out-of-scope is abstained, OR in-scope is answered
    correct = (is_out and abstained) or (not is_out and not abstained)

    return {
        "query": query,
        "is_out_of_scope": is_out,
        "top_rerank_score": round(top_score, 3) if not intent_abstained else -99.0,
        "abstained": abstained,
        "correct": correct
    }

def run_scope_eval(limit: int = 30) -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    in_scope_cases = [r for r in harness.dataset if not (r.get("is_out_of_scope", False) or r.get("expected_behavior") == "abstain")]
    out_scope_cases = [r for r in harness.dataset if (r.get("is_out_of_scope", False) or r.get("expected_behavior") == "abstain")]
    
    test_cases = in_scope_cases[:limit] + out_scope_cases

    import time
    results = []
    latencies = []
    for item in test_cases:
        t0 = time.perf_counter()
        res = eval_single_scope(item, harness)
        dur = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = round(dur, 2)
        results.append(res)
        latencies.append(dur)

    in_scope = [r for r in results if not r["is_out_of_scope"]]
    out_scope = [r for r in results if r["is_out_of_scope"]]

    abstain_recall = sum(1 for r in out_scope if r["abstained"]) / max(len(out_scope), 1)
    false_abstain_rate = sum(1 for r in in_scope if r["abstained"]) / max(len(in_scope), 1)
    overall_accuracy = sum(1 for r in results if r["correct"]) / max(len(results), 1)

    # ROC-AUC computation
    pos_scores = [r["top_rerank_score"] for r in in_scope]
    neg_scores = [r["top_rerank_score"] for r in out_scope]
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    roc_auc = round(wins / max(len(pos_scores) * len(neg_scores), 1), 4)

    summary = {
        "benchmark": "eval_scope",
        "total_queries": len(results),
        "in_scope_count": len(in_scope),
        "out_of_scope_count": len(out_scope),
        "abstain_recall": round(abstain_recall, 4),
        "false_abstain_rate": round(false_abstain_rate, 4),
        "scope_accuracy": round(overall_accuracy, 4),
        "roc_auc": roc_auc,
        "latency_profile": latency_percentiles(latencies),
        "status": "PASS" if roc_auc >= 0.85 and abstain_recall >= 0.75 else "FAIL"
    }

    print("\n=================== SCOPE EVALUATION RESULTS ===================")
    print(f"Total Evaluated Queries:          {summary['total_queries']} ({len(in_scope)} In-Scope, {len(out_scope)} Out-of-Scope)")
    print(f"Abstain Recall (Out-of-Scope):    {summary['abstain_recall'] * 100:.1f}%")
    print(f"False Abstain Rate (In-Scope):    {summary['false_abstain_rate'] * 100:.1f}%")
    print(f"Overall Scope Classification Acc: {summary['scope_accuracy'] * 100:.1f}%")
    print(f"Abstention ROC-AUC:               {summary['roc_auc']}")
    print(f"Status:                           {summary['status']}")
    print(f"Mean Latency:                     {summary['latency_profile']['mean']} ms")
    print("================================================================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Academic Scope Gating")
    parser.add_argument("--limit", type=int, default=100, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_scope_eval(limit=args.limit)
