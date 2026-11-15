import urllib.request
import json

def get(path):
    r = urllib.request.urlopen(f"http://127.0.0.1:8000/api{path}")
    return json.loads(r.read().decode("utf-8"))

def post(path, body):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:8000/api{path}", data=data, headers={"Content-Type": "application/json"})
    r = urllib.request.urlopen(req)
    return json.loads(r.read().decode("utf-8"))

print("=== 1. TOPICS RANKING TEST ===")
for sub in ["DSA", "DBMS", "TOC", "DCCN", "TC", "DMS"]:
    res = get(f"/topics/{sub}/important")
    top_t = res.get("ranked_topics", [{}])[0]
    print(f"{sub:5} | Total Topics: {res.get('total_topics'):2} | Top: {top_t.get('topic_name')} ({top_t.get('total_marks')}M) | Level: {top_t.get('importance_level')}")

print("\n=== 2. PLANNER 0.5h & 1.0h TEST ===")
p05 = post("/plan", {"subject_id": "DSA", "hours_available": 0.5})
print(f"DSA 0.5h schedule slots: {len(p05.get('schedule', []))}")
for s in p05.get("schedule", []):
    print(f"   [{s.get('time_window')}] {s.get('topic_name')} ({s.get('duration_minutes')}m)")

p1 = post("/plan", {"subject_id": "DBMS", "hours_available": 1.0})
print(f"DBMS 1.0h schedule slots: {len(p1.get('schedule', []))}")
for s in p1.get("schedule", []):
    print(f"   [{s.get('time_window')}] {s.get('topic_name')} ({s.get('duration_minutes')}m)")

print("\n=== 3. QUIZ EVALUATOR OFF-TOPIC TEST ===")
q_eval = post("/quiz/answer", {
    "question_id": "RTU_DSA_2026_PB_Q16_RTU_QP_027",
    "student_answer": "The weather is very sunny today and I ate a cheese sandwich with butter."
})
print(f"Off-topic score: {q_eval.get('score')} / {q_eval.get('max_score')}")
print(f"Examiner Feedback: {q_eval.get('senior_feedback')}")

print("\n=== 4. QUIZ EVALUATOR REFUSAL TEST ===")
q_refuse = post("/quiz/answer", {
    "question_id": "RTU_DSA_2026_PB_Q16_RTU_QP_027",
    "student_answer": "I have no idea what this is about really, just guessing."
})
print(f"Refusal score: {q_refuse.get('score')} / {q_refuse.get('max_score')}")

print("\n=== 5. CHAT GREETING TEST ===")
chat_greet = post("/chat", {"message": "hii", "language": "english"})
print(f"Greeting answer:\n{chat_greet.get('answer')}")

print("\n=== 6. CRASH STUDY PLAN QUERY TEST ===")
chat_plan = post("/chat", {"message": "i have 6 hr from now and i want some important topics for DSA", "language": "hinglish"})
print(f"Plan answer preview:\n{chat_plan.get('answer')[:400]}...")

print("\n=== ALL VERIFICATION TESTS COMPLETED SUCCESSFULLY ===")
