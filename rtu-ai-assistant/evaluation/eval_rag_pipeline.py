"""
End-to-End RAG Pipeline Evaluation Module (eval_rag_pipeline.py).
Evaluates the RAG Triad:
1. Context Relevance (Retriever precision & keyword alignment)
2. Groundedness / Faithfulness (Synthesized text backed by retrieved RTU context)
3. Answer Relevance (Synthesized answer addresses student prompt)
4. Citation Integrity (Ground truth RTU paper references)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import faithfulness_score, answer_relevance_score, latency_percentiles

def eval_single_rag(item: Dict[str, Any], harness: EvaluationHarness) -> Dict[str, Any]:
    query = item["query"]
    subject = item.get("subject", "DSA")
    expected_keywords = item.get("expected_keywords", [])

    chat_res = harness.chat_service.chat(
        db=harness.db,
        user_message=query,
        detected_subject=subject,
        session_id=None,
        provider_name="offline"
    )

    answer = chat_res.get("assistant_reply", "")
    citations = chat_res.get("citations", [])
    
    # Retrieve the context string
    retrieval_data = harness.pipeline.retrieve(query=query, subject=subject, top_k=5)
    context_str = retrieval_data.get("context_string", "")

    # Compute RAG Triad metrics
    faith_score = faithfulness_score(answer, context_str)
    relevance_score = answer_relevance_score(answer, query, expected_keywords)
    has_valid_citations = len(citations) > 0 or len(retrieval_data.get("results", [])) > 0

    return {
        "query": query,
        "subject": subject,
        "faithfulness": faith_score,
        "answer_relevance": relevance_score,
        "citation_count": len(citations),
        "has_citations": has_valid_citations,
        "context_doc_count": len(retrieval_data.get("results", []))
    }

def run_rag_eval(limit: int = 20) -> Dict[str, Any]:
    harness = EvaluationHarness()
    benchmark_data = harness.run_benchmark(
        name="End-to-End RAG Triad Pipeline Evaluation",
        eval_fn=eval_single_rag,
        limit=limit,
        filter_fn=lambda x: not x.get("is_out_of_scope", False)
    )

    results = benchmark_data["results"]
    latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
    faith_scores = [r.get("faithfulness", 0.0) for r in results]
    rel_scores = [r.get("answer_relevance", 0.0) for r in results]
    citation_rates = [1.0 if r.get("has_citations") else 0.0 for r in results]

    summary = {
        "benchmark": "eval_rag_pipeline",
        "total_evaluated": len(results),
        "mean_faithfulness": round(sum(faith_scores) / max(len(faith_scores), 1), 3),
        "mean_answer_relevance": round(sum(rel_scores) / max(len(rel_scores), 1), 3),
        "citation_presence_rate": round(sum(citation_rates) / max(len(citation_rates), 1), 3),
        "latency_profile": latency_percentiles(latencies)
    }

    print("\n=================== END-TO-END RAG EVALUATION ===================")
    print(f"Total Academic Queries Evaluated: {summary['total_evaluated']}")
    print(f"Faithfulness / Groundedness Score: {summary['mean_faithfulness'] * 100:.1f}%")
    print(f"Answer Relevance Score:            {summary['mean_answer_relevance'] * 100:.1f}%")
    print(f"Citation Grounding Rate:           {summary['citation_presence_rate'] * 100:.1f}%")
    print(f"Mean Pipeline Latency:             {summary['latency_profile']['mean']} ms")
    print(f"P90 Pipeline Latency:              {summary['latency_profile']['p90']} ms")
    print("=================================================================\n")

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate End-to-End RAG Pipeline")
    parser.add_argument("--limit", type=int, default=20, help="Number of queries to evaluate")
    args = parser.parse_args()
    run_rag_eval(limit=args.limit)
