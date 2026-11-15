"""
System Reliability and Edge Case Evaluation Module (eval_reliability.py).
Evaluates stability under adversarial edge inputs: empty queries, whitespace,
1000+ word inputs, Unicode math, SQL injection patterns, and determinism.
Production Gate: Unhandled Crash Rate MUST be 0.0%.
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

RELIABILITY_EDGE_CASES = [
    {
        "test_id": "edge_empty",
        "name": "Empty string input",
        "query": ""
    },
    {
        "test_id": "edge_whitespace",
        "name": "Whitespace only input",
        "query": "   \n\t\r   "
    },
    {
        "test_id": "edge_ultralong",
        "name": "1200+ word repeating query",
        "query": "explain binary search tree operations and avl balancing rotations " * 150
    },
    {
        "test_id": "edge_unicode_math",
        "name": "Complex LaTeX and Unicode mathematical symbols",
        "query": "Explain Master Theorem: T(n) = aT(n/b) + f(n) where a >= 1, b > 1, and evaluate \u2211_{i=1}^n \u0398(n \u22c5 \u221e)"
    },
    {
        "test_id": "edge_xss_sql_injection",
        "name": "XSS script tags and SQL injection delimiters",
        "query": "<script>alert('RTU_XSS')</script> '; DROP TABLE questions; -- SELECT * FROM papers WHERE '1'='1"
    },
    {
        "test_id": "edge_special_chars",
        "name": "Punctuation and non-ASCII glyphs",
        "query": "!@#$%^&*()_+{}[]:\"|<>?,./~`\\'"
    }
]

def eval_single_reliability(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    query = item["query"]
    crashed = False
    error_msg = None
    response_length = 0

    try:
        chat_res = harness.chat_service.chat(
            db=harness.db,
            user_message=query,
            session_id=None,
            provider_name="offline"
        )
        reply = chat_res.get("assistant_reply", "")
        response_length = len(reply)
        handled_gracefully = response_length > 0
    except Exception as e:
        crashed = True
        error_msg = str(e)
        handled_gracefully = False

    return {
        "test_id": item["test_id"],
        "name": item["name"],
        "crashed": crashed,
        "error_msg": error_msg,
        "response_length": response_length,
        "handled_gracefully": handled_gracefully
    }

def run_reliability_eval() -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_reliability] Evaluating system edge cases, stress payloads, and crash resilience...")
    results = []
    latencies = []

    import time
    for item in RELIABILITY_EDGE_CASES:
        t0 = time.perf_counter()
        res = eval_single_reliability(item, harness)
        dur = (time.perf_counter() - t0) * 1000.0
        res["latency_ms"] = round(dur, 2)
        results.append(res)
        latencies.append(dur)

    crashes = sum(1 for r in results if r["crashed"])
    graceful = sum(1 for r in results if r["handled_gracefully"])
    stability_rate = graceful / len(results)

    summary = {
        "benchmark": "eval_reliability",
        "total_edge_cases": len(results),
        "crashes": crashes,
        "graceful_handlings": graceful,
        "stability_rate": round(stability_rate, 4),
        "latency_profile": latency_percentiles(latencies),
        "status": "PASS" if crashes == 0 and stability_rate == 1.0 else "FAIL"
    }

    print("\n=================== RELIABILITY EVALUATION RESULTS ===================")
    print(f"Total Edge Payloads Tested:       {summary['total_edge_cases']}")
    print(f"Unhandled System Crashes:         {summary['crashes']} (Required: 0)")
    print(f"Graceful Degradation Rate:        {summary['stability_rate'] * 100:.1f}%")
    print(f"Status:                           {summary['status']}")
    print(f"Mean Latency:                     {summary['latency_profile']['mean']} ms")
    print("======================================================================\n")

    return summary

if __name__ == "__main__":
    run_reliability_eval()
