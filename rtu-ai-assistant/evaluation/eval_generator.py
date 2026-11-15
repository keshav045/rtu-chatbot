"""
Generator Evaluation Module (eval_generator.py).
Evaluates LLM generation against RTU 6-Point Academic Rubric:
1. Technical Definition
2. Architecture / ASCII Diagram
3. Step-by-Step Algorithm / Derivation
4. Working Example with traced values
5. Time & Space Complexity Big-O notation
6. RTU Examiner Scoring Tips
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import rtu_rubric_adherence_score, latency_percentiles

def eval_single_generation(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    query = item["query"]
    subject = item.get("subject", "DSA")

    # Generate answer through ChatService
    chat_res = harness.chat_service.chat(
        db=harness.db,
        user_message=query,
        detected_subject=subject,
        session_id=None,
        provider_name="offline"
    )

    answer = chat_res.get("assistant_reply", "")
    rubric = rtu_rubric_adherence_score(answer)

    return {
        "query": query,
        "subject": subject,
        "answer_length": len(answer),
        "citations_count": len(chat_res.get("citations", [])),
        "rubric_score": rubric["score"],
        "rubric_breakdown": rubric["breakdown"],
        "has_definition": rubric["breakdown"]["definition"],
        "has_diagram": rubric["breakdown"]["diagram"],
        "has_steps": rubric["breakdown"]["steps"],
        "has_example": rubric["breakdown"]["example"],
        "has_complexity": rubric["breakdown"]["complexity"],
        "has_scoring_tips": rubric["breakdown"]["scoring_tips"]
    }

def run_generator_eval(limit: int = 20) -> Dict[str, Any]:
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="RTU Academic Answer Generator Evaluation",
        eval_fn=eval_single_generation,
        limit=limit,
        filter_fn=lambda x: not x.get("is_out_of_scope", False)
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    scores = [r.get("rubric_score", 0) for r in results]
    def_rates = [1.0 if r.get("has_definition") else 0.0 for r in results]
    diag_rates = [1.0 if r.get("has_diagram") else 0.0 for r in results]
    step_rates = [1.0 if r.get("has_steps") else 0.0 for r in results]
    ex_rates = [1.0 if r.get("has_example") else 0.0 for r in results]
    comp_rates = [1.0 if r.get("has_complexity") else 0.0 for r in results]
    tip_rates = [1.0 if r.get("has_scoring_tips") else 0.0 for r in results]

    summary = {
        "benchmark": "eval_generator",
        "total_evaluated": len(results),
        "mean_rubric_score": round(sum(scores) / max(len(scores), 1), 2),
        "definition_rate": round(sum(def_rates) / max(len(def_rates), 1), 3),
        "diagram_rate": round(sum(diag_rates) / max(len(diag_rates), 1), 3),
        "steps_rate": round(sum(step_rates) / max(len(step_rates), 1), 3),
        "example_rate": round(sum(ex_rates) / max(len(ex_rates), 1), 3),
        "complexity_rate": round(sum(comp_rates) / max(len(comp_rates), 1), 3),
        "scoring_tips_rate": round(sum(tip_rates) / max(len(tip_rates), 1), 3),
        "latency_profile": latency_percentiles(latencies)
    }

    print("\n=================== GENERATOR EVALUATION RESULTS ===================")
    print(f"Total Academic Queries Evaluated: {summary['total_evaluated']}")
    print(f"Mean RTU Rubric Score (out of 6): {summary['mean_rubric_score']} / 6.0")
    print(f"Definition Presence Rate:         {summary['definition_rate'] * 100:.1f}%")
    print(f"Diagram / Structure Rate:         {summary['diagram_rate'] * 100:.1f}%")
    print(f"Algorithm / Steps Rate:           {summary['steps_rate'] * 100:.1f}%")
    print(f"Worked Example Rate:              {summary['example_rate'] * 100:.1f}%")
    print(f"Big-O Complexity Rate:            {summary['complexity_rate'] * 100:.1f}%")
    print(f"Scoring Tips Rate:                {summary['scoring_tips_rate'] * 100:.1f}%")
    print(f"Mean Latency:                     {summary['latency_profile']['mean']} ms")
    print("====================================================================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Academic Answer Generation")
    parser.add_argument("--limit", type=int, default=20, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_generator_eval(limit=args.limit)
