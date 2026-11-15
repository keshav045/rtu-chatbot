"""
Benchmark runner v2 for the RTU AI Assistant (works with eval_dataset_v2_100.json).

Put both files in  rtu-ai-assistant/evaluation/  and run from the project root:

    python evaluation/run_eval_v2.py                       # retrieval, no subject filter
    python evaluation/run_eval_v2.py --with-subject        # retrieval, subject filter on
    python evaluation/run_eval_v2.py --threshold -7        # abstention threshold to test
    python evaluation/run_eval_v2.py --chat-url http://127.0.0.1:8000/api   # also test /chat end to end

Metrics
  In-scope (85 queries)
    Hit@1, Hit@5, Hit@10    at least one relevant question in the top k
    Precision@5             share of the top 5 that are relevant
    Recall@5, Recall@10     relevant found / min(k, number of relevant questions that exist)
    MRR                     mean reciprocal rank of the first relevant question
  Abstention (15 queries: 8 out-of-scope + 7 subjects with no data)
    Abstain recall          share of "should abstain" queries correctly rejected
    False-abstain rate      share of in-scope queries wrongly rejected
    ROC-AUC                 how well the top rerank score separates in-scope from abstain queries
    Best threshold          threshold that maximises (abstain recall - false-abstain rate)

A question is "relevant" when it belongs to the expected subject AND contains at least
`min_keyword_hits` of the expected keywords. Matching ignores case, spaces and punctuation
because the OCR text often has missing spaces ("Findthevolume").
"""

import re
import sys
import json
import time
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
BACKEND_DIR = HERE.parent / "backend"
DATASET = HERE / "eval_dataset_v2_100.json"
OUTPUT = HERE / "benchmark_results_v2.json"

KS = (1, 5, 10)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def keyword_hits(text: str, keywords) -> int:
    t = norm(text)
    return sum(1 for k in keywords if k and k in t)


def is_relevant(doc: dict, case: dict) -> bool:
    if case.get("subject") and doc.get("subject_short", "").upper() != case["subject"].upper():
        return False
    return keyword_hits(doc.get("question_text", ""), case["expected_keywords"]) >= case.get("min_keyword_hits", 1)


def relevant_ids(doc_store: dict, case: dict) -> set:
    return {qid for qid, d in doc_store.items() if is_relevant(d, case)}


def roc_auc(pos_scores, neg_scores) -> float:
    """P(score of an in-scope query > score of an abstain query); ties count 0.5."""
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos_scores) * len(neg_scores))


def percentile(values, q):
    if not values:
        return 0.0
    v = sorted(values)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def evaluate(retrieve_fn, doc_store, cases, with_subject=False, threshold=-10.0, warmup=True):
    """
    retrieve_fn(query, subject, top_k) -> {"results": [ {question_id, rerank_score, subject_short, question_text, ...} ]}
    doc_store: {question_id: doc dict} (pipeline.doc_store)
    """
    if warmup and cases:
        try:
            retrieve_fn("warm up query", None, 3)   # first call loads the models; keep it out of the timing
        except Exception:
            pass

    rows, latencies = [], []
    for tc in cases:
        subject = tc.get("subject") if (with_subject and not tc["is_out_of_scope"]) else None
        t0 = time.perf_counter()
        res = retrieve_fn(tc["query"], subject, max(KS))
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)
        results = res.get("results", [])
        top_score = results[0].get("rerank_score", -999.0) if results else -999.0
        row = {
            "query_id": tc["query_id"], "query": tc["query"], "category": tc["category"],
            "subject": tc.get("subject"), "expected_behavior": tc["expected_behavior"],
            "top_score": round(top_score, 3), "latency_ms": round(ms, 1),
            "top_subject": results[0].get("subject_short") if results else None,
            "top_source": results[0].get("source_pdf") if results else None,
            "abstained": top_score < threshold,
        }
        if not tc["is_out_of_scope"]:
            R = relevant_ids(doc_store, tc)
            ids = [r["question_id"] for r in results]
            flags = [qid in R for qid in ids]
            first = next((i + 1 for i, f in enumerate(flags) if f), None)
            row.update({
                "relevant_in_corpus": len(R),
                "rank_first_relevant": first,
                "reciprocal_rank": (1.0 / first) if first else 0.0,
                "precision_at_5": (sum(flags[:5]) / max(1, min(5, len(flags)))) if flags else 0.0,
            })
            for k in KS:
                row[f"hit_at_{k}"] = int(any(flags[:k]))
                row[f"recall_at_{k}"] = (sum(flags[:k]) / min(k, len(R))) if R else 0.0
        rows.append(row)

    ins = [r for r in rows if r["expected_behavior"] == "answer"]
    aba = [r for r in rows if r["expected_behavior"] == "abstain"]
    mean = lambda key, xs: round(sum(x[key] for x in xs) / len(xs), 4) if xs else None

    metrics = {"n_in_scope": len(ins), "n_abstain_expected": len(aba), "threshold": threshold}
    for k in KS:
        metrics[f"hit_at_{k}"] = mean(f"hit_at_{k}", ins)
    metrics["precision_at_5"] = mean("precision_at_5", ins)
    metrics["recall_at_5"] = mean("recall_at_5", ins)
    metrics["recall_at_10"] = mean("recall_at_10", ins)
    metrics["mrr"] = mean("reciprocal_rank", ins)
    metrics["abstain_recall"] = round(sum(r["abstained"] for r in aba) / len(aba), 4) if aba else None
    metrics["false_abstain_rate"] = round(sum(r["abstained"] for r in ins) / len(ins), 4) if ins else None
    pos = [r["top_score"] for r in ins]
    neg = [r["top_score"] for r in aba]
    metrics["abstain_roc_auc"] = round(roc_auc(pos, neg), 4)

    # threshold sweep (uses every observed score as a candidate threshold)
    best = None
    for th in sorted(set(pos + neg)):
        rec = sum(s < th for s in neg) / len(neg) if neg else 0
        fa = sum(s < th for s in pos) / len(pos) if pos else 0
        j = rec - fa
        if best is None or j > best["youden_j"]:
            best = {"threshold": round(th, 3), "abstain_recall": round(rec, 3), "false_abstain_rate": round(fa, 3), "youden_j": round(j, 3)}
    metrics["best_threshold"] = best

    metrics["latency_ms_p50"] = round(statistics.median(latencies), 1) if latencies else 0
    metrics["latency_ms_p95"] = round(percentile(latencies, 0.95), 1)

    def group(key):
        g = defaultdict(list)
        for r in ins:
            g[r[key]].append(r)
        return {k: {"n": len(v), "hit_at_1": mean("hit_at_1", v), "hit_at_5": mean("hit_at_5", v),
                    "recall_at_5": mean("recall_at_5", v), "mrr": mean("reciprocal_rank", v)}
                for k, v in sorted(g.items(), key=lambda kv: str(kv[0]))}

    return {"metrics": metrics, "by_category": group("category"), "by_subject": group("subject"), "rows": rows}


def chat_check(base_url, cases, refusal_regex):
    """Optional end-to-end check against a running server (needs `pip install httpx`)."""
    import httpx
    refusal = re.compile(refusal_regex, re.I)
    out = []
    with httpx.Client(timeout=120) as c:
        for tc in cases:
            try:
                r = c.post(f"{base_url}/chat", json={"message": tc["query"], "provider": "offline"})
                data = r.json() if r.status_code == 200 else {}
                ans = data.get("answer", "")
                cites = data.get("citations", [])
                if tc["expected_behavior"] == "answer":
                    ok = r.status_code == 200 and any(
                        (ci.get("subject") or "").upper() == tc["subject"].upper() for ci in cites)
                else:
                    ok = r.status_code == 200 and (bool(refusal.search(ans)) or not cites)
                out.append({"query_id": tc["query_id"], "status": r.status_code, "pass": ok})
            except Exception as e:
                out.append({"query_id": tc["query_id"], "status": "error", "pass": False, "error": str(e)})
    n = len(out)
    return {"chat_pass_rate": round(sum(o["pass"] for o in out) / n, 4) if n else None, "details": out}


def print_report(rep):
    m = rep["metrics"]
    line = "=" * 74
    print(f"\n{line}\nRTU AI ASSISTANT: BENCHMARK v2 SUMMARY\n{line}")
    print(f"In-scope queries: {m['n_in_scope']}   Should-abstain queries: {m['n_abstain_expected']}")
    print(f"Hit@1 {m['hit_at_1']:.1%} | Hit@5 {m['hit_at_5']:.1%} | Hit@10 {m['hit_at_10']:.1%}")
    print(f"Precision@5 {m['precision_at_5']:.1%} | Recall@5 {m['recall_at_5']:.1%} | Recall@10 {m['recall_at_10']:.1%} | MRR {m['mrr']:.3f}")
    print(f"Abstain @ threshold {m['threshold']}: recall {m['abstain_recall']:.1%}, false-abstain {m['false_abstain_rate']:.1%}, ROC-AUC {m['abstain_roc_auc']}")
    print(f"Best threshold on this run: {m['best_threshold']}")
    print(f"Latency p50 {m['latency_ms_p50']} ms | p95 {m['latency_ms_p95']} ms (warm-up excluded)")
    print("\nBy category (in-scope):")
    for k, v in rep["by_category"].items():
        print(f"  {k:14s} n={v['n']:2d}  Hit@1 {v['hit_at_1']:.0%}  Hit@5 {v['hit_at_5']:.0%}  Recall@5 {v['recall_at_5']:.0%}  MRR {v['mrr']:.2f}")
    print("\nBy subject (in-scope):")
    for k, v in rep["by_subject"].items():
        print(f"  {k:6s} n={v['n']:2d}  Hit@1 {v['hit_at_1']:.0%}  Hit@5 {v['hit_at_5']:.0%}  Recall@5 {v['recall_at_5']:.0%}  MRR {v['mrr']:.2f}")
    misses = [r for r in rep["rows"] if r["expected_behavior"] == "answer" and not r["hit_at_5"]]
    print(f"\nIn-scope queries with NO relevant result in top 5: {len(misses)}")
    for r in misses[:15]:
        print(f"  {r['query_id']} [{r['subject']}] {r['query'][:60]} -> top: {r['top_subject']} ({r['top_score']})")
    bad = [r for r in rep["rows"] if r["expected_behavior"] == "abstain" and not r["abstained"]]
    print(f"\nShould-abstain queries that were NOT rejected: {len(bad)}")
    for r in bad:
        print(f"  {r['query_id']} [{r['category']}] {r['query'][:55]} -> score {r['top_score']}, top subject {r['top_subject']}")
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(DATASET))
    ap.add_argument("--with-subject", action="store_true", help="pass the expected subject as a retrieval filter")
    ap.add_argument("--threshold", type=float, default=None, help="abstain when the top rerank score is below this (default: the value used in chat_service.py)")
    ap.add_argument("--chat-url", default=None, help="e.g. http://127.0.0.1:8000/api to also run an end-to-end /chat check")
    ap.add_argument("--refusal-regex", default=r"no relevant|nahi mil raha|match nahi|not found|couldn't find|can't find|notice")
    ap.add_argument("--output", default=str(OUTPUT))
    args = ap.parse_args()

    sys.path.insert(0, str(BACKEND_DIR))
    from app.database.session import SessionLocal
    from app.retrieval.pipeline import RetrievalPipeline
    from app.core.config import settings

    cases = json.load(open(args.dataset, encoding="utf-8"))
    db = SessionLocal()
    pipe = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
    pipe.load_indexes(db)

    def retrieve_fn(q, subject, k):
        return pipe.retrieve(query=q, subject=subject, top_k=k)

    th = args.threshold if args.threshold is not None else -10.0
    rep = evaluate(retrieve_fn, pipe.doc_store, cases, with_subject=args.with_subject, threshold=th)
    if args.chat_url:
        rep["chat"] = chat_check(args.chat_url, cases, args.refusal_regex)
        print(f"/chat end-to-end pass rate: {rep['chat']['chat_pass_rate']}")
    print_report(rep)
    rep["config"] = {"with_subject": args.with_subject, "threshold": th, "dataset": Path(args.dataset).name,
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(rep, open(args.output, "w", encoding="utf-8"), indent=2)
    print(f"Full results written to {args.output}")
    db.close()


if __name__ == "__main__":
    main()
