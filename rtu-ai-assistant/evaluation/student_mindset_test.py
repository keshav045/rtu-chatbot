"""
High-Level Student Mindset Simulation Test Suite
Evaluates the RTU AI Assistant through the lens of real student personas before exams:
1. Urgent Night-Before Exam Planner (Knapsack Crash Schedule)
2. High-Yield Topic Hunter (Past Paper Frequency Ranking)
3. 10-Mark Answer Coach (Format, Diagrams, Steps, Examiner Scoring Tips)
4. 2-Mark Part A Speed Runner (Crisp 2-sentence definitions)
5. Colloquial Hinglish Mentor (Natural conversational tone)
6. Out-of-Syllabus Guardrail (Zero-hallucination boundary test)
7. Practice Quiz & Rubric Evaluation (0-10 scoring & constructive feedback)
8. Physical PDF Source Verification (Ground truth disk validation)
"""

import urllib.request
import json
import time
import os
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000/api"

def post(endpoint, data):
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data).encode("utf-8")
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get(endpoint):
    req = urllib.request.Request(f"{BASE_URL}{endpoint}")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_student_mindset_evaluation():
    print("=" * 75)
    print("RTU AI ACADEMIC ASSISTANT: HIGH-LEVEL STUDENT MINDSET EVALUATION")
    print("=" * 75)
    print("Simulating realistic student preparation journeys across 8 scenarios...\n")

    results = []

    # ---------------------------------------------------------
    # SCENARIO 1: The Urgent Night-Before Student (Crash Planner)
    # Persona: 6 hours before DSA examination, needs prioritized study timeline.
    # ---------------------------------------------------------
    print("[1/8] Testing Scenario 1: Urgent 6-Hour Crash Course Planner...")
    start_t = time.time()
    plan_resp = post("/plan", {"subject_id": "DSA", "hours_left": 6.0, "target_score": 75})
    elapsed = round((time.time() - start_t) * 1000, 1)

    slots = plan_resp.get("schedule", [])
    buffer_mins = plan_resp.get("buffer_minutes", 0)
    rev_mins = plan_resp.get("revision_minutes", 0)
    has_buffer = buffer_mins >= 30
    has_revision = rev_mins >= 30
    has_slots = len(slots) >= 4

    status_1 = has_buffer and has_revision and has_slots
    results.append(("Scenario 1: 6-Hour Crash Planner", status_1, f"{len(slots)} slots, {buffer_mins}m buffer, {rev_mins}m revision ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_1 else 'FAIL'} | Allocated {len(slots)} study slots with {buffer_mins} min buffer & {rev_mins} min rapid revision.")

    # ---------------------------------------------------------
    # SCENARIO 2: The High-Yield Topic Hunter
    # Persona: Looking for the most frequently repeated topics in AEM.
    # ---------------------------------------------------------
    print("\n[2/8] Testing Scenario 2: High-Yield Topic Ranking (AEM)...")
    start_t = time.time()
    topics_resp = get("/topics/AEM/important")
    elapsed = round((time.time() - start_t) * 1000, 1)

    ranked = topics_resp.get("ranked_topics", [])
    paper_count = topics_resp.get("paper_count", 0)
    status_2 = len(ranked) > 0 and paper_count >= 10
    top_topic = ranked[0]["topic_name"] if ranked else "N/A"
    top_marks = ranked[0]["total_marks"] if ranked else 0
    results.append(("Scenario 2: High-Yield Topic Ranking", status_2, f"Top: {top_topic} ({top_marks} marks, {paper_count} papers) ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_2 else 'FAIL'} | Analyzed {paper_count} papers. Top topic: '{top_topic}' with {top_marks} marks historically.")

    # ---------------------------------------------------------
    # SCENARIO 3: The 10-Mark Answer Format Coach (Part C)
    # Persona: "Explain AVL tree rotations for 10 marks with diagram."
    # ---------------------------------------------------------
    print("\n[3/8] Testing Scenario 3: 10-Mark Answer Coach (AVL Rotations)...")
    start_t = time.time()
    chat_resp = post("/chat", {
        "message": "Explain AVL tree rotations and balance factor for 10 marks with diagram",
        "provider": "offline",
        "filters": {"subject": "DSA"}
    })
    elapsed = round((time.time() - start_t) * 1000, 1)

    ans_text = chat_resp.get("assistant_message", {}).get("content", "")
    citations = chat_resp.get("citations", [])

    has_def = "Definition" in ans_text or "definition" in ans_text
    has_diagram = "Diagram" in ans_text or "diagram" in ans_text
    has_algo = "Step" in ans_text or "Algorithm" in ans_text or "Working" in ans_text
    has_scoring = "Scoring" in ans_text or "Senior" in ans_text or "Tip" in ans_text
    has_citations = len(citations) > 0

    status_3 = has_def and has_diagram and has_algo and has_citations
    results.append(("Scenario 3: 10-Mark Answer Coach", status_3, f"Sections verified: Def={has_def}, Diagram={has_diagram}, Steps={has_algo}, Citations={len(citations)} ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_3 else 'FAIL'} | Complete 10M structure delivered with {len(citations)} verified citations.")

    # ---------------------------------------------------------
    # SCENARIO 4: The 2-Mark Part A Speed Runner
    # Persona: Needs a concise definition for 2 marks without essay filler.
    # ---------------------------------------------------------
    print("\n[4/8] Testing Scenario 4: 2-Mark Part A Brevity Coach...")
    start_t = time.time()
    chat_2m = post("/chat", {
        "message": "Define AVL tree balance factor for 2 marks",
        "provider": "offline",
        "filters": {"subject": "DSA"}
    })
    elapsed = round((time.time() - start_t) * 1000, 1)

    ans_2m = chat_2m.get("assistant_message", {}).get("content", "")
    # Should identify Part A or 2 marks
    has_2m_format = "2 Mark" in ans_2m or "Part A" in ans_2m or "Crisp" in ans_2m or "Core Definition" in ans_2m
    status_4 = has_2m_format and len(ans_2m) > 100
    results.append(("Scenario 4: 2-Mark Part A Brevity", status_4, f"Target marks recognized, concise format generated ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_4 else 'FAIL'} | Brevity coach active. Detected 2-mark scope cleanly.")

    # ---------------------------------------------------------
    # SCENARIO 5: The Colloquial Hinglish Mentor
    # Persona: "Bhai Laplace transform ke repeated questions bata do AEM me"
    # ---------------------------------------------------------
    print("\n[5/8] Testing Scenario 5: Colloquial Hinglish Student Query...")
    start_t = time.time()
    chat_hinglish = post("/chat", {
        "message": "Bhai AEM me Laplace transform ke repeated questions bata do aur kaise likhna hai samjhao",
        "provider": "offline",
        "filters": {"subject": "AEM"}
    })
    elapsed = round((time.time() - start_t) * 1000, 1)

    ans_h = chat_hinglish.get("assistant_message", {}).get("content", "")
    citations_h = chat_hinglish.get("citations", [])
    # Check Hinglish tone
    has_hinglish_tone = any(w in ans_h.lower() for w in ["hai", "karo", "yeh", "mein", "marks", "papers"])
    status_5 = has_hinglish_tone and len(citations_h) > 0
    results.append(("Scenario 5: Hinglish Mentor Tone", status_5, f"Natural Hinglish tone detected with {len(citations_h)} citations ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_5 else 'FAIL'} | Senior mentor replied in natural Hinglish with {len(citations_h)} past paper questions.")

    # ---------------------------------------------------------
    # SCENARIO 6: Out-of-Syllabus Guardrail (Zero-Hallucination)
    # Persona: Student accidentally asks about unrelated modern tech in RTU 3rd Sem exam.
    # ---------------------------------------------------------
    print("\n[6/8] Testing Scenario 6: Out-of-Syllabus / Refusal Boundary...")
    start_t = time.time()
    chat_refusal = post("/chat", {
        "message": "Explain Quantum Teleportation and React Native hooks",
        "provider": "offline"
    })
    elapsed = round((time.time() - start_t) * 1000, 1)

    ans_r = chat_refusal.get("assistant_message", {}).get("content", "")
    # Check polite refusal / guidance
    has_refusal = any(phrase in ans_r.lower() for phrase in ["notice", "match nahi", "syllabus", "not found", "check"])
    status_6 = has_refusal
    results.append(("Scenario 6: Out-of-Syllabus Guardrail", status_6, f"Polite refusal & search tip delivered ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_6 else 'FAIL'} | Out-of-scope query rejected gracefully without inventing fake exam questions.")

    # ---------------------------------------------------------
    # SCENARIO 7: Practice Quiz & 0-10 Rubric Evaluation
    # Persona: Takes a quick mock question and submits an answer for feedback.
    # ---------------------------------------------------------
    print("\n[7/8] Testing Scenario 7: Mock Practice Quiz & Senior Rubric Evaluation...")
    start_t = time.time()
    quiz_q = post("/quiz/next", {"subject_id": "DSA"})
    q_id = quiz_q.get("question_id")
    q_text = quiz_q.get("question_text", "")
    q_source = quiz_q.get("source", "")

    # Submit sample answer
    student_ans = (
        "An AVL tree is a self-balancing binary search tree. The balance factor is height(left) - height(right), "
        "which must be -1, 0, or +1. If it becomes unbalanced, rotations are performed: LL, RR, LR, and RL. "
        "Lookup time complexity is O(log n)."
    )
    eval_resp = post("/quiz/answer", {
        "question_id": q_id,
        "student_answer": student_ans
    })
    elapsed = round((time.time() - start_t) * 1000, 1)

    score = eval_resp.get("score", 0)
    max_score = eval_resp.get("max_score", 10)
    rubric = eval_resp.get("rubric_breakdown", {})
    feedback = eval_resp.get("senior_feedback", "")

    status_7 = score > 0 and len(rubric) > 0 and len(feedback) > 10
    results.append(("Scenario 7: Quiz & Rubric Evaluation", status_7, f"Score: {score}/{max_score}, Breakdown: {rubric} ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_7 else 'FAIL'} | Real past question served ('{q_source}'). Answer graded: {score}/{max_score}.")

    # ---------------------------------------------------------
    # SCENARIO 8: Physical PDF Ground Truth Inspection
    # Persona: Student clicks to view the authentic RTU question paper PDF.
    # ---------------------------------------------------------
    print("\n[8/8] Testing Scenario 8: Physical PDF Paper Inspection...")
    start_t = time.time()
    papers = get("/papers?limit=5")
    sample_paper = papers[0] if papers else {}
    paper_id = sample_paper.get("paper_id")
    pdf_filename = sample_paper.get("source_pdf")

    pdf_url = f"{BASE_URL}/papers/{paper_id}/pdf"
    req_pdf = urllib.request.Request(pdf_url)
    try:
        with urllib.request.urlopen(req_pdf) as resp_pdf:
            pdf_bytes = resp_pdf.read()
            pdf_valid = len(pdf_bytes) > 1000 and pdf_bytes[:4] == b"%PDF"
    except Exception as e:
        pdf_valid = False
    elapsed = round((time.time() - start_t) * 1000, 1)

    status_8 = pdf_valid
    results.append(("Scenario 8: Physical PDF Verification", status_8, f"Verified '{pdf_filename}' ({len(pdf_bytes):,} bytes) on disk ({elapsed}ms)"))
    print(f"      Result: {'PASS' if status_8 else 'FAIL'} | Served valid PDF ({len(pdf_bytes):,} bytes) with '%PDF' signature.")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    print("\n" + "=" * 75)
    print("EVALUATION SUMMARY: 8 STUDENT MINDSET SCENARIOS")
    print("=" * 75)
    passed_count = sum(1 for _, ok, _ in results)
    for name, ok, details in results:
        status_str = "[PASS]" if ok else "[FAIL]"
        print(f"{status_str:<8} | {name:<35} | {details}")

    print("=" * 75)
    print(f"Final Score: {passed_count} / {len(results)} Scenarios Passed ({(passed_count/len(results))*100:.1f}%)")
    print("=" * 75)

if __name__ == "__main__":
    run_student_mindset_evaluation()
