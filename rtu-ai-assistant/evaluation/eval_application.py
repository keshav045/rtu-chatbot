"""
Application User Journey Evaluation Module (eval_application.py).
Evaluates full student user workflows across all 4 core platform modules:
1. Senior Tutor Interactive Chat (`ChatService`)
2. Crash Course Hour-by-Hour Planner (`PlannerService`)
3. High-Yield Topic Frequency & Importance Rankings (`ImportanceService`)
4. Authentic Exam Practice Quiz & Answer Evaluation (`QuizService`)
Production Gate: 100% Student Journey Pass Rate across all 4 workflows.
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

def eval_chat_journey(harness: EvaluationHarness) -> Dict[str, Any]:
    t0 = time.perf_counter()
    res = harness.chat_service.chat(
        db=harness.db,
        user_message="Explain Binary Search algorithm with time complexity and recurrence",
        detected_subject="DSA",
        provider_name="offline"
    )
    dur = (time.perf_counter() - t0) * 1000.0
    reply = res.get("assistant_reply", "")
    citations = res.get("citations", [])
    passed = len(reply) > 100 and ("algorithm" in reply.lower() or "binary" in reply.lower())

    return {
        "module": "Senior Tutor Chat",
        "passed": passed,
        "latency_ms": round(dur, 2),
        "reply_length": len(reply),
        "citation_count": len(citations)
    }

def eval_planner_journey(harness: EvaluationHarness) -> Dict[str, Any]:
    from app.services.planner_service import PlannerService
    planner = PlannerService()
    t0 = time.perf_counter()
    plan = planner.generate_plan(db=harness.db, subject_id="DSA", hours_available=6.0)
    dur = (time.perf_counter() - t0) * 1000.0
    
    schedule = plan.get("schedule", plan.get("study_blocks", []))
    passed = len(schedule) > 0

    return {
        "module": "Crash Course Planner",
        "passed": passed,
        "latency_ms": round(dur, 2),
        "total_study_blocks": len(schedule),
        "total_allocated_minutes": plan.get("allocated_study_minutes", plan.get("total_minutes_scheduled", 0))
    }

def eval_importance_journey(harness: EvaluationHarness) -> Dict[str, Any]:
    from app.services.importance_service import ImportanceService
    importance = ImportanceService()
    t0 = time.perf_counter()
    ranking = importance.get_ranked_topics(db=harness.db, subject_id="DSA")
    dur = (time.perf_counter() - t0) * 1000.0

    topics = ranking.get("ranked_topics", [])
    passed = len(topics) > 0 and ("importance_level" in topics[0] or "total_marks" in topics[0])

    return {
        "module": "Topic Importance Rankings",
        "passed": passed,
        "latency_ms": round(dur, 2),
        "ranked_topics_count": len(topics),
        "top_topic": topics[0].get("topic_name") if topics else None
    }

def eval_quiz_journey(harness: EvaluationHarness) -> Dict[str, Any]:
    from app.services.quiz_service import QuizService
    quiz_svc = QuizService()
    t0 = time.perf_counter()
    q = quiz_svc.get_next_question(db=harness.db, subject_id="DSA")
    dur = (time.perf_counter() - t0) * 1000.0

    passed = "question_text" in q and len(q["question_text"]) > 10

    return {
        "module": "Practice Quiz",
        "passed": passed,
        "latency_ms": round(dur, 2),
        "question_id": q.get("question_id"),
        "marks": q.get("marks")
    }

def run_application_eval() -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_application] Running integration tests across all 4 student feature workflows...")

    journeys = [
        eval_chat_journey(harness),
        eval_planner_journey(harness),
        eval_importance_journey(harness),
        eval_quiz_journey(harness)
    ]

    all_passed = all(j["passed"] for j in journeys)
    latencies = [j["latency_ms"] for j in journeys]

    summary = {
        "benchmark": "eval_application",
        "total_workflows": len(journeys),
        "workflows_passed": sum(1 for j in journeys if j["passed"]),
        "pass_rate": round(sum(1 for j in journeys if j["passed"]) / len(journeys), 4),
        "workflows": journeys,
        "latency_profile": latency_percentiles(latencies),
        "status": "PASS" if all_passed else "FAIL"
    }

    print("\n=================== APPLICATION WORKFLOW RESULTS ===================")
    for j in journeys:
        status_text = "PASS" if j["passed"] else "FAIL"
        print(f"[{status_text}] {j['module']:<28} Latency: {j['latency_ms']:6.1f} ms")
    print("-------------------------------------------------------------------")
    print(f"Overall Workflow Pass Rate: {summary['pass_rate'] * 100:.1f}%")
    print(f"Status:                     {summary['status']}")
    print(f"Mean Workflow Latency:      {summary['latency_profile']['mean']} ms")
    print("===================================================================\n")

    return summary

if __name__ == "__main__":
    run_application_eval()
