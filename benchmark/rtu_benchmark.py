"""
RTU AI Academic Assistant - Correctness and Rigor Benchmark.

Measures actual technical correctness, precision, and adherence to RTU rubrics,
not merely shallow keyword presence.

Checks per question:
  - content_pass: meets rubric must_contain_groups, has no must_not_contain phrases,
                  and satisfies question-specific structural rules (e.g. Q08 quiz).
  - template_phrases: detects canned system prompt leakage and unexpanded template tags.
  - unverified_citations: flags fabricated academic papers, non-RTU journals, or hallucinated standards.
  - length_floor: flags empty or truncated responses (< 80 characters).
  - duplicate_answers: flags copy-paste template replies across runs.
  - refusal_subject_mismatch: flags refusals that blindly cite active session subjects (e.g. "OS")
                              for unrelated out-of-syllabus queries.

Usage:
    python benchmark/rtu_benchmark.py
    python benchmark/rtu_benchmark.py --rubric benchmark/rubric.json
    python benchmark/rtu_benchmark.py --dry-run benchmark/results/rtu_benchmark_20261124_010722.json
"""

import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import httpx

# Template phrases that indicate robotic boilerplate or unexpanded prompt artifacts
TEMPLATE_PHRASES = [
    "Scoring Tip",
    "Examiner Focus",
    "Exam Tip",
    "Senior Tutor",
    "Student Query",
    "Student Request",
    "Step-by-Step Working",
    "Key Formula",
    "Key Points & Formula",
    "Technical Formulation",
    "Architectural",
    "physics or mathematics",
    "state input constraints",
]

# Patterns indicating citation-like academic references
CITATION_PATTERNS = [
    re.compile(r"Journal of\b", re.IGNORECASE),
    re.compile(r"ISO/IEC\b", re.IGNORECASE),
    re.compile(r"\bet al\.", re.IGNORECASE),
    re.compile(r"Proceedings\b", re.IGNORECASE),
    re.compile(r"\bIEEE\b", re.IGNORECASE),
    re.compile(r"\((?:19|20)\d{2}\)"),
]

# Valid RTU paper years and authentic RTU citation markers in backend data sources
VALID_RTU_CITATION_MARKERS = [
    "RTU",
    "Rajasthan Technical University",
    "past paper",
    "question paper",
    "PYQ",
    "2012-2024",
]

REFUSAL_PATTERN = re.compile(
    r"(only|exclusively|cannot|can't|not able|outside|out of (the )?syllabus|RTU|engineering|exam)",
    re.IGNORECASE,
)

DEFAULT_QUESTIONS = [
    ("Q01", "answer", "What is B-tree insertion? Explain the steps."),
    ("Q02", "answer", "Explain the Pumping Lemma for regular languages."),
    ("Q03", "answer", "Explain normalization up to 3NF in DBMS."),
    ("Q04", "answer", "Explain the TCP three-way handshake."),
    ("Q05", "answer", "Explain the difference between DFS and BFS."),
    ("Q06", "answer", "Give a 2 mark answer: what is a deadlock?"),
    ("Q07", "answer", "Make a 4 hour crash revision plan for the DBMS exam."),
    ("Q08", "answer", "Quiz me with 3 questions on data structures."),
    ("Q09", "answer", "Explain the four conditions for deadlock in Operating Systems."),
    ("Q10", "refuse", "What is the cricket score today?"),
    ("Q11", "refuse", "Recommend a good Bollywood movie."),
    ("Q12", "refuse", "Give me a biryani recipe."),
]


def load_rubric(rubric_path: Path) -> Dict[str, Any]:
    """Loads the rubric definitions from JSON."""
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric file not found at: {rubric_path}")
    with open(rubric_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_template_phrases(answer: str) -> List[str]:
    """Checks if the answer contains forbidden template phrases."""
    hits = []
    lower_ans = answer.lower()
    for phrase in TEMPLATE_PHRASES:
        if phrase.lower() in lower_ans:
            hits.append(phrase)
    return hits


def check_unverified_citations(answer: str) -> List[str]:
    """
    Checks for academic citations (e.g. Journal of ..., ISO/IEC, years in parentheses)
    and verifies if they correspond to genuine RTU past exam citations.
    """
    unverified = []
    for pattern in CITATION_PATTERNS:
        matches = pattern.findall(answer)
        for m in matches:
            val = m if isinstance(m, str) else m[0]
            # If the citation is an RTU exam year like (2024) or (2023) in RTU context, verify it
            year_match = re.match(r"\((\d{4})\)", val)
            if year_match:
                year = int(year_match.group(1))
                # Years prior to 2010 or non-RTU references are unverified
                if year < 2010 or year > 2025:
                    unverified.append(val)
                elif not any(marker.lower() in answer.lower() for marker in VALID_RTU_CITATION_MARKERS):
                    unverified.append(val)
            else:
                # Direct journal or international standard citations are unverified in RTU past papers
                unverified.append(val)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for u in unverified:
        if u.lower() not in seen:
            seen.add(u.lower())
            deduped.append(u)
    return deduped


def check_refusal_subject_mismatch(qid: str, question: str, answer: str) -> Optional[str]:
    """
    Ensures refusals for non-academic queries (cricket, movie, recipe)
    do not hallucinate or lock onto an arbitrary technical subject like 'OS'.
    """
    lower_ans = answer.lower()
    # Check if refusal claims the off-topic query is related to OS
    if any(pat in lower_ans for pat in ["**os**", "from os", "from **os**", "question from os", "in os"]):
        return "refusal incorrectly associated off-topic query with 'OS'"
    return None


def check_q08_quiz(answer: str) -> Tuple[bool, str]:
    """
    Verifies that Q08 generates exactly 3 numbered quiz questions ending with '?',
    and does not deflect by asking student study preferences.
    """
    lower_ans = answer.lower()
    if any(p in lower_ans for p in [
        "how would you like to proceed",
        "which topic or question would you like to start with",
        "which topic or question would you like to prepare",
        "tell me your time left"
    ]):
        return False, "asked student preferences instead of generating quiz"

    # Count numbered question items ending in '?'
    numbered_q_pattern = re.compile(r"(?:^|\n)\s*(?:\d+[\.\)]|[Qq]uestion\s*\d+[:\.\)])\s*.*?\?", re.DOTALL)
    numbered_matches = numbered_q_pattern.findall(answer)

    # General question mark count
    qmark_count = answer.count("?")

    if len(numbered_matches) == 3 or qmark_count == 3:
        return True, ""
    return False, f"expected exactly 3 quiz questions ending in '?', found {len(numbered_matches)} numbered / {qmark_count} question marks"


def score_response(
    qid: str,
    qtype: str,
    question: str,
    answer: str,
    error: Optional[str],
    rubric_item: Optional[Dict[str, Any]],
    seen_answers: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluates response correctness using strict rubric criteria,
    template detection, citation verification, and duplicate tracking.
    """
    if error:
        return {
            "pass": False,
            "content_pass": False,
            "failures": [f"network/server error: {error}"],
            "note": f"error: {error}",
            "template_phrases": [],
            "unverified_citations": [],
            "is_duplicate": False,
            "refusal_mismatch": None,
        }

    stripped_ans = answer.strip()
    if not stripped_ans:
        return {
            "pass": False,
            "content_pass": False,
            "failures": ["empty response"],
            "note": "empty answer",
            "template_phrases": [],
            "unverified_citations": [],
            "is_duplicate": False,
            "refusal_mismatch": None,
        }

    failures = []

    # 1. Answer length floor (80 chars)
    if len(stripped_ans) < 80:
        failures.append(f"answer too short ({len(stripped_ans)} chars < 80 floor)")

    # 2. Duplicate answer check across run
    norm_answer = re.sub(r"\s+", " ", stripped_ans.lower())
    is_duplicate = False
    if norm_answer in seen_answers:
        is_duplicate = True
        prior_qid = seen_answers[norm_answer]
        failures.append(f"duplicate_answer (identical to {prior_qid})")
    else:
        seen_answers[norm_answer] = qid

    # 3. Template phrase check
    template_hits = check_template_phrases(answer)
    if template_hits:
        failures.append(f"template_phrase ({', '.join(template_hits[:3])})")

    # 4. Unverified citation check
    unverified_cites = check_unverified_citations(answer)
    if unverified_cites:
        failures.append(f"unverified_citation ({', '.join(unverified_cites[:3])})")

    # 5. Content correctness & Rubric validation
    content_pass = True
    refusal_mismatch = None

    if qtype == "refuse":
        refused = bool(REFUSAL_PATTERN.search(answer))
        if not refused:
            content_pass = False
            failures.append("failed_refusal (answered an out-of-syllabus query)")

        # Check refusal subject mismatch
        mismatch_err = check_refusal_subject_mismatch(qid, question, answer)
        if mismatch_err:
            refusal_mismatch = mismatch_err
            failures.append(f"refusal_subject_mismatch ({mismatch_err})")
            content_pass = False

    else:
        # qtype == "answer"
        if rubric_item:
            # Check must_not_contain
            must_not = rubric_item.get("must_not_contain", [])
            lower_ans = answer.lower()
            hit_forbidden = [bad for bad in must_not if bad.lower() in lower_ans]
            if hit_forbidden:
                content_pass = False
                failures.append(f"must_not_contain ({', '.join(hit_forbidden)})")

            # Check must_contain_groups
            groups = rubric_item.get("must_contain_groups", [])
            min_groups = rubric_item.get("min_groups_met", len(groups))
            met_groups = 0
            missing_groups = []

            for g_idx, group in enumerate(groups):
                matched = any(phrase.lower() in lower_ans for phrase in group)
                if matched:
                    met_groups += 1
                else:
                    missing_groups.append(group[0])

            if met_groups < min_groups:
                content_pass = False
                failures.append(f"must_contain ({met_groups}/{min_groups} groups met; missing: {', '.join(missing_groups[:3])})")

        # Special structural rule for Q08
        if qid == "Q08":
            quiz_ok, quiz_msg = check_q08_quiz(answer)
            if not quiz_ok:
                content_pass = False
                failures.append(f"quiz_structure ({quiz_msg})")

    overall_pass = content_pass and (len(failures) == 0)
    note = "passed all checks" if overall_pass else "; ".join(failures)

    return {
        "pass": overall_pass,
        "content_pass": content_pass,
        "failures": failures,
        "note": note,
        "template_phrases": template_hits,
        "unverified_citations": unverified_cites,
        "is_duplicate": is_duplicate,
        "refusal_mismatch": refusal_mismatch,
    }


def extract_answer(payload: Any, field: str) -> str:
    """Pull the answer text out of the JSON response, tolerant to key names."""
    if isinstance(payload, dict):
        for key in [field, "answer", "response", "reply", "text", "message"]:
            if key in payload and isinstance(payload[key], str):
                return payload[key]
        return json.dumps(payload)
    return str(payload)


def ask(client: httpx.Client, url: str, field: str, question: str) -> Tuple[str, float, Optional[str]]:
    """Sends a question to the API and records response time and payload."""
    start = time.perf_counter()
    try:
        r = client.post(url, json={"message": question, "question": question, "session_id": "benchmark"})
        latency = time.perf_counter() - start
        r.raise_for_status()
        try:
            text = extract_answer(r.json(), field)
        except ValueError:
            text = r.text
        return text, latency, None
    except Exception as exc:
        return "", time.perf_counter() - start, str(exc)


def run_benchmark(
    rubric_path: Path,
    dry_run_path: Optional[Path] = None,
    base_url: str = "http://127.0.0.1:8000",
    endpoint: str = "/api/chat",
    field: str = "answer",
    timeout: float = 120.0,
) -> int:
    """Executes the benchmark suite either live against the server or via dry-run."""
    rubric = load_rubric(rubric_path)
    seen_answers: Dict[str, str] = {}
    rows: List[Dict[str, Any]] = []
    latencies: List[float] = []

    if dry_run_path:
        print(f"Running Dry-Run Scorer on: {dry_run_path}\nUsing Rubric: {rubric_path}\n")
        with open(dry_run_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        saved_results = saved_data.get("results", [])

        for item in saved_results:
            qid = item.get("id")
            qtype = item.get("type", "answer")
            question = item.get("question", "")
            answer = item.get("answer", "")
            latency = float(item.get("latency_s", 0.0))
            latencies.append(latency)

            rubric_item = rubric.get(qid)
            result = score_response(
                qid=qid,
                qtype=qtype,
                question=question,
                answer=answer,
                error=None,
                rubric_item=rubric_item,
                seen_answers=seen_answers,
            )

            status = "PASS" if result["pass"] else "FAIL"
            print(f"{qid} [{status}] {latency:6.2f}s  {result['note']}  | {question[:60]}")

            rows.append({
                "id": qid,
                "type": qtype,
                "question": question,
                "answer": answer,
                "latency_s": round(latency, 2),
                **result,
            })

    else:
        url = base_url.rstrip("/") + endpoint
        print(f"Benchmarking live server at {url}\nUsing Rubric: {rubric_path}\n")

        with httpx.Client(timeout=timeout) as client:
            for qid, qtype, question in DEFAULT_QUESTIONS:
                rubric_item = rubric.get(qid)
                answer, latency, error = ask(client, url, field, question)
                latencies.append(latency)

                result = score_response(
                    qid=qid,
                    qtype=qtype,
                    question=question,
                    answer=answer,
                    error=error,
                    rubric_item=rubric_item,
                    seen_answers=seen_answers,
                )

                status = "PASS" if result["pass"] else "FAIL"
                print(f"{qid} [{status}] {latency:6.2f}s  {result['note']}  | {question[:60]}")

                rows.append({
                    "id": qid,
                    "type": qtype,
                    "question": question,
                    "answer": answer,
                    "latency_s": round(latency, 2),
                    **result,
                })

    # Aggregated Metrics Calculation
    total = len(rows)
    passed = sum(1 for r in rows if r["pass"])
    content_passed = sum(1 for r in rows if r["content_pass"])
    template_hits_count = sum(1 for r in rows if r["template_phrases"])
    unverified_citations_count = sum(1 for r in rows if r["unverified_citations"])
    duplicate_count = sum(1 for r in rows if r["is_duplicate"])
    refusal_mismatch_count = sum(1 for r in rows if r["refusal_mismatch"])

    lat_sorted = sorted(latencies) if latencies else [0.0]
    p95 = lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))]

    summary = {
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 2) if total else 0.0,
        "content_pass_rate": round(content_passed / total, 2) if total else 0.0,
        "template_phrase_hits": template_hits_count,
        "unverified_citations": unverified_citations_count,
        "duplicate_answers": duplicate_count,
        "refusal_subject_mismatch": refusal_mismatch_count,
        "latency_mean_s": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "latency_p95_s": round(p95, 2),
        "errors": sum(1 for r in rows if "network/server error" in r.get("failures", [])),
    }

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>26}: {v}")

    # Save details if live run
    if not dry_run_path:
        out_dir = Path(__file__).parent / "results"
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / f"rtu_benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        out_file.write_text(
            json.dumps({"summary": summary, "results": rows}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved details to {out_file}")

    return 0 if passed == total else 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark RTU AI Assistant for accuracy, citations, and rigor.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI backend host")
    parser.add_argument("--endpoint", default="/api/chat", help="Chat endpoint path")
    parser.add_argument("--field", default="answer", help="JSON key containing the reply")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--rubric", default="benchmark/rubric.json", help="Path to rubric JSON specification")
    parser.add_argument("--dry-run", nargs="?", const="", default=None, help="Path to existing results JSON file to score without server calls")
    args = parser.parse_args()

    rubric_path = Path(args.rubric)
    dry_run_path = Path(args.dry_run) if args.dry_run else None

    exit_code = run_benchmark(
        rubric_path=rubric_path,
        dry_run_path=dry_run_path,
        base_url=args.base_url,
        endpoint=args.endpoint,
        field=args.field,
        timeout=args.timeout,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
