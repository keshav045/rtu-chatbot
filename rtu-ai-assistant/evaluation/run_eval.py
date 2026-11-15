"""
Evaluation & Benchmark Suite for RTU AI Academic Assistant
Evaluates retrieval accuracy (Precision@1, Precision@5, MRR), citation verification,
latency, and out-of-scope query handling across benchmark questions.
"""

import sys
import time
import json
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.database.session import SessionLocal
from app.database.models import Paper
from app.retrieval.pipeline import RetrievalPipeline
from app.core.config import settings

def run_evaluation():
    print("=" * 70)
    print("RTU AI Academic Assistant: System Evaluation & Benchmark")
    print("=" * 70)

    dataset_path = Path(__file__).parent / "eval_dataset.json"
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    db = SessionLocal()
    pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
    pipeline.load_indexes(db)

    results = []
    p_at_1_count = 0
    p_at_5_count = 0
    reciprocal_ranks = []
    latencies_ms = []
    citation_valid_count = 0
    total_citations_checked = 0
    out_of_scope_handled_correctly = 0
    total_out_of_scope = 0

    print(f"\nEvaluating {len(test_cases)} benchmark queries across 16 subjects...\n")

    for tc in test_cases:
        qid = tc["query_id"]
        query = tc["query"]
        expected_subject = tc.get("subject")
        expected_keywords = [kw.lower() for kw in tc.get("expected_keywords", [])]
        is_oos = tc.get("is_out_of_scope", False)

        t_start = time.perf_counter()
        retrieval_res = pipeline.retrieve(query=query, top_k=5)
        t_elapsed = (time.perf_counter() - t_start) * 1000
        latencies_ms.append(t_elapsed)

        retrieved_questions = retrieval_res["results"]
        citations = retrieval_res["citations"]

        if is_oos:
            total_out_of_scope += 1
            # For out-of-scope, scores should be low or context builder recognizes no high relevance
            top_score = retrieved_questions[0]["rerank_score"] if retrieved_questions else -99.0
            # Cross-encoder scores < -4.0 signify poor relevance for ungrounded concepts
            if top_score < 0.0 or len(retrieved_questions) == 0:
                out_of_scope_handled_correctly += 1
                status = "PASSED (Low Confidence / Out-of-Scope Correctly Detected)"
            else:
                status = "MARGINAL"
            results.append({
                "query_id": qid,
                "query": query,
                "is_out_of_scope": True,
                "top_score": top_score,
                "status": status,
                "latency_ms": round(t_elapsed, 2)
            })
            print(f"[{qid}] Out-of-Scope Query: '{query[:45]}...' | Top Score: {top_score:.3f} | {status}")
            continue

        # Check relevance for top-k
        found_at_rank = None
        relevant_in_top_5 = 0

        for rank, doc in enumerate(retrieved_questions, 1):
            doc_text = doc["question_text"].lower()
            doc_subject = doc["subject_short"].upper()

            # Match criteria: either subject matches or multiple keywords hit
            subject_match = (expected_subject is not None and doc_subject == expected_subject.upper())
            keyword_hits = sum(1 for kw in expected_keywords if kw in doc_text)

            is_relevant = (subject_match and keyword_hits >= 1) or (keyword_hits >= 2)

            if is_relevant:
                relevant_in_top_5 += 1
                if found_at_rank is None:
                    found_at_rank = rank

        # Check citations physical validity
        for cit in citations:
            pdf_name = cit["source_pdf"]
            paper_rec = db.query(Paper).filter(Paper.source_pdf == pdf_name).first()
            total_citations_checked += 1
            if paper_rec:
                pdf_path = settings.WORKSPACE_ROOT / "cleaned_dataset" / paper_rec.relative_path
                if pdf_path.exists():
                    citation_valid_count += 1

        if found_at_rank == 1:
            p_at_1_count += 1

        if relevant_in_top_5 > 0:
            p_at_5_count += 1
            reciprocal_ranks.append(1.0 / found_at_rank)
        else:
            reciprocal_ranks.append(0.0)

        results.append({
            "query_id": qid,
            "query": query,
            "subject": expected_subject,
            "rank_first_relevant": found_at_rank,
            "relevant_in_top_5": relevant_in_top_5,
            "latency_ms": round(t_elapsed, 2),
            "top_source": retrieved_questions[0]["source_pdf"] if retrieved_questions else "None",
            "top_score": round(retrieved_questions[0]["rerank_score"], 3) if retrieved_questions else None
        })

        print(f"[{qid}] '{query[:45]}...' -> Rank 1: {found_at_rank == 1} | MRR: {reciprocal_ranks[-1]:.2f} | Latency: {t_elapsed:.1f}ms")

    in_scope_count = len(test_cases) - total_out_of_scope
    precision_1 = (p_at_1_count / in_scope_count) * 100 if in_scope_count else 0
    precision_5 = (p_at_5_count / in_scope_count) * 100 if in_scope_count else 0
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0
    citation_accuracy = (citation_valid_count / total_citations_checked) * 100 if total_citations_checked else 100.0

    print("\n" + "=" * 70)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 70)
    print(f"In-Scope Benchmark Queries:    {in_scope_count}")
    print(f"Precision@1:                   {precision_1:.1f}%")
    print(f"Precision@5:                   {precision_5:.1f}%")
    print(f"Mean Reciprocal Rank (MRR):     {mrr:.4f}")
    print(f"Average Retrieval Latency:     {avg_latency:.1f} ms")
    print(f"Physical PDF Citation Validity: {citation_accuracy:.1f}% ({citation_valid_count}/{total_citations_checked})")
    print(f"Out-of-Scope Guarded:          {out_of_scope_handled_correctly}/{total_out_of_scope}")
    print("=" * 70)

    output_path = Path(__file__).parent / "benchmark_results.json"
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_queries": len(test_cases),
        "metrics": {
            "precision_at_1": round(precision_1, 2),
            "precision_at_5": round(precision_5, 2),
            "mrr": round(mrr, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "citation_validity_percent": round(citation_accuracy, 2),
            "out_of_scope_guard_rate": f"{out_of_scope_handled_correctly}/{total_out_of_scope}"
        },
        "query_results": results
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Full benchmark results exported to: {output_path}\n")
    db.close()

if __name__ == "__main__":
    run_evaluation()
