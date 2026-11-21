"""
Expanded Fine-Tuning Dataset Generator for RTU Senior Tutor
Generates a comprehensive dataset of 600-1000+ training pairs in standard ChatML JSONL format
leveraging 1,534 authentic RTU past examination questions across Semesters 3 to 8.

Covers the 5 core tasks:
1. explain_topic (10-mark structured answers & 2-mark brevity in English & Hinglish)
2. quiz_generation (5-question multi-choice and short-answer practice drills with syllabus citations)
3. answer_grading (0-10 scoring rubric breakdown with constructive feedback)
4. structured_extraction (Raw OCR paper text to structured JSON schema)
5. refusal_idk (Zero hallucination guardrails for out-of-syllabus and non-academic queries)
"""

import os
import sys
import json
import random
import sqlite3
from pathlib import Path

# Set random seed for reproducible dataset synthesis
random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "backend" / "data" / "storage" / "rtu_metadata.db"
SEED_FILE = BASE_DIR / "seed_dataset_200.jsonl"
OUTPUT_FILE = BASE_DIR / "rtu_train_dataset_expanded.jsonl"

SYSTEM_PROMPT = (
    "You are Senior Tutor - an empathetic, high-scoring senior student from Rajasthan Technical University (RTU). "
    "You explain concepts in simple English or Hinglish, provide exam-oriented 10-mark structures (Definition, Diagram, Steps, Example, RTU Tip), "
    "grade student answers honestly with constructive feedback, and strictly refuse out-of-syllabus topics."
)

def build_dataset():
    data = []

    # 1. Load existing seed dataset if available
    if SEED_FILE.exists():
        with open(SEED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        data.append(record)
                    except Exception:
                        pass
        print(f"Loaded {len(data)} foundational examples from {SEED_FILE.name}")

    # 2. Connect to RTU metadata database
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Query questions with metadata
    query_questions = """
        SELECT q.id, q.text, q.marks, q.part, p.year, s.short_name, s.name, u.unit_no, u.title
        FROM questions q
        JOIN papers p ON q.paper_id = p.id
        JOIN subjects s ON p.subject_id = s.id
        LEFT JOIN topics t ON q.topic_id = t.id
        LEFT JOIN units u ON t.unit_id = u.id
        WHERE length(q.text) > 30
        ORDER BY s.short_name, p.year DESC
    """
    rows = cursor.execute(query_questions).fetchall()
    print(f"Retrieved {len(rows)} authentic questions from rtu_metadata.db")

    # Group questions by subject
    subj_questions = {}
    for r in rows:
        qid, qtext, qmarks, qpart, qyear, s_short, s_name, unit_no, unit_title = r
        if s_short not in subj_questions:
            subj_questions[s_short] = []
        subj_questions[s_short].append({
            "id": qid,
            "text": qtext.strip(),
            "marks": qmarks or (15 if "C" in (qpart or "") else (8 if "B" in (qpart or "") else 2)),
            "part": qpart or "Part B",
            "year": qyear,
            "subject_code": s_short,
            "subject_name": s_name,
            "unit_no": unit_no or 1,
            "unit_title": unit_title or "Core Concepts"
        })

    # Subject specific knowledge banks for detailed 10-mark expansions
    subject_details = {
        "DSA": {
            "diagram": "Draw memory layout / pointer links or recursion tree clearly labeled.",
            "formula": "Time Complexity: O(log n) to O(n log n); Recurrence: T(n) = 2T(n/2) + O(n)",
            "rtu_tip": "Always write standard C/C++ struct definitions and state Best, Average, and Worst case complexities!"
        },
        "DBMS": {
            "diagram": "Draw Relational schema table rectangles with primary/foreign key connector arrows, or B+ tree node pointers.",
            "formula": "Closure computation F+ and Armstrong's Axioms (Reflexivity, Augmentation, Transitivity).",
            "rtu_tip": "Examiner expects complete SQL syntax or Relational Algebra symbols. Underline Primary Keys!"
        },
        "TOC": {
            "diagram": "Draw State Transition Diagram with double circles for final states and initial start arrow.",
            "formula": "5-tuple formal notation: M = (Q, Sigma, delta, q0, F).",
            "rtu_tip": "Draw the complete state transition table alongside the state diagram for full marks."
        },
        "DCCN": {
            "diagram": "Draw 7-layer OSI / 5-layer TCP/IP stack diagram with header encapsulation arrows.",
            "formula": "Efficiency = 1 / (1 + 2a) where a = Propagation Delay / Transmission Delay.",
            "rtu_tip": "Write header byte sizes (e.g., IPv4 20-60 bytes, TCP 20-60 bytes, UDP 8 bytes)."
        },
        "OS": {
            "diagram": "Draw Process State Transition Diagram (New, Ready, Running, Waiting, Terminated) or Paging MMU architecture.",
            "formula": "Banker's Algorithm: Need[i][j] = Max[i][j] - Allocation[i][j].",
            "rtu_tip": "Include Gantt chart for CPU scheduling numericals and show step-by-step Turnaround/Waiting times."
        },
        "DAA": {
            "diagram": "Draw divide-and-conquer recursion tree or dynamic programming memoization matrix.",
            "formula": "Master Theorem: T(n) = aT(n/b) + f(n); compare f(n) with n^(log_b a).",
            "rtu_tip": "Always write pseudocode with line numbers and give space/time asymptotic bounds."
        },
        "CD": {
            "diagram": "Draw 6-phase compiler flow diagram (Lexical -> Syntax -> Semantic -> ICG -> Optimization -> Target Code).",
            "formula": "FIRST and FOLLOW set calculation rules, canonical collection of LR(0) items.",
            "rtu_tip": "Draw complete parsing tables with Action and Goto sections clearly partitioned."
        },
        "AI": {
            "diagram": "Draw Search Tree / State Space graph with heuristic cost values h(n) and path costs g(n).",
            "formula": "A* Evaluation Function: f(n) = g(n) + h(n), where h(n) is admissible (never overestimates).",
            "rtu_tip": "Explain admissibility and consistency conditions explicitly in Part C answers."
        },
        "MI": {
            "diagram": "Draw 8085/8086 microprocessor internal architecture block diagram with ALU, registers, and bus interfaces.",
            "formula": "Clock frequency f = crystal / 2; Machine cycle T-states calculation.",
            "rtu_tip": "Show T-state timing diagrams with ALE, IO/M, RD, WR signal levels labeled."
        },
        "DE": {
            "diagram": "Draw logic gate schematics with truth tables and Karnaugh Map (K-map) groupings.",
            "formula": "Boolean algebra identities: De Morgan's laws and canonical SOP / POS forms.",
            "rtu_tip": "Always draw K-map cell coordinates (00, 01, 11, 10) in Gray code sequence."
        },
        "DMS": {
            "diagram": "Draw Hasse diagram for Poset, Venn diagrams, or Graph adjacency representation.",
            "formula": "Pigeonhole Principle, Principle of Inclusion-Exclusion, and Generating Functions.",
            "rtu_tip": "Show formal induction proof steps (Base Step, Inductive Hypothesis, Inductive Step)."
        },
        "AEM": {
            "diagram": "Draw coordinate axes or probability distribution bell curves where applicable.",
            "formula": "Taylor/Maclaurin series, Laplace transforms L{f(t)}, and Numerical method formulas.",
            "rtu_tip": "Write the general formula first before substituting numerical values; box the final answer."
        },
        "RTS": {
            "diagram": "Draw Periodic Task Execution Timeline with Arrival, Deadline, and Execution slices.",
            "formula": "Liu & Layland Utilization Bound for Rate Monotonic: U = sum(C_i / T_i) <= n(2^(1/n) - 1).",
            "rtu_tip": "Distinguish clearly between Hard, Soft, and Firm real-time deadline guarantees."
        },
        "DS": {
            "diagram": "Draw Distributed System Architecture: Client-Server, Peer-to-Peer, or Message Passing Rings.",
            "formula": "Lamport's Logical Clocks: C(b) = max(C(b), C(a) + 1).",
            "rtu_tip": "Explain CAP theorem (Consistency, Availability, Partition tolerance) trade-offs."
        },
        "MC": {
            "diagram": "Draw Cellular frequency reuse cluster geometry (N=7 or N=4) and Mobile IP handover signaling.",
            "formula": "Cluster size N = i^2 + ij + j^2; Signal-to-Interference Ratio (SIR).",
            "rtu_tip": "Mention GSM architecture entities: MS, BTS, BSC, MSC, HLR, VLR, AUC, EIR."
        },
        "BDA": {
            "diagram": "Draw Hadoop HDFS Master-Worker architecture (NameNode & DataNodes) and MapReduce execution flow.",
            "formula": "Map: (k1, v1) -> list(k2, v2); Reduce: (k2, list(v2)) -> list(k3, v3).",
            "rtu_tip": "Discuss the 5 V's of Big Data: Volume, Velocity, Variety, Veracity, Value."
        },
        "IOT": {
            "diagram": "Draw 4-layer IoT architecture: Sensing / Device -> Network -> Service -> Application layer.",
            "formula": "MQTT publish/subscribe topic hierarchy and QoS levels (0: At most once, 1: At least once, 2: Exactly once).",
            "rtu_tip": "Highlight low-power wireless protocols: Zigbee (802.15.4), LoRaWAN, BLE, CoAP."
        },
        "ISS": {
            "diagram": "Draw Symmetric vs Asymmetric encryption pipeline and Public Key Infrastructure (PKI) flow.",
            "formula": "RSA Key Generation: n = p*q, phi(n) = (p-1)*(q-1), e*d = 1 mod phi(n).",
            "rtu_tip": "State the CIA Triad: Confidentiality, Integrity, Availability with relevant security controls."
        },
        "TC": {
            "diagram": "Draw Shannon-Weaver communication model diagram (Sender -> Encoding -> Channel -> Decoding -> Receiver).",
            "formula": "7 C's of Effective Communication: Clear, Concise, Concrete, Correct, Coherent, Complete, Courteous.",
            "rtu_tip": "Use professional headings, formal memo formats, and standard resume templates."
        },
        "MEFA": {
            "diagram": "Draw Break-Even Chart showing Fixed Cost, Variable Cost, Total Cost, Total Revenue, and Break-Even Point.",
            "formula": "Break-Even Point (Units) = Fixed Cost / (Selling Price per unit - Variable Cost per unit).",
            "rtu_tip": "Define Law of Demand with assumptions, and clearly list factors causing shifts vs movements."
        },
        "OOP": {
            "diagram": "Draw Class Diagram showing Inheritance hierarchy (base class -> derived class) and Encapsulation boundaries.",
            "formula": "Syntax for Virtual Table (vptr & vtable) and operator overloading signatures.",
            "rtu_tip": "Contrast Compile-time (Overloading) vs Run-time Polymorphism (Overriding with virtual functions)."
        },
        "SE": {
            "diagram": "Draw SDLC Life Cycle / Spiral Model with risk analysis sectors, or DFD Level 0 and Level 1.",
            "formula": "COCOMO Model: Effort E = a * (KLOC)^b person-months; Cyclomatic Complexity V(G) = E - N + 2P.",
            "rtu_tip": "Always specify validation vs verification, and contrast Waterfall vs Agile Scrum sprints."
        }
    }

    # ==========================================================================
    # TASK 1: explain_topic (10-Mark Model Answers & 2-Mark Crisp Definitions)
    # ==========================================================================
    explain_count = 0
    for subj_code, qlist in subj_questions.items():
        spec = subject_details.get(subj_code, {
            "diagram": "Draw neat block diagram with clearly labeled inputs, processes, and outputs.",
            "formula": "Mathematical formulation and standard algorithmic complexity bounds.",
            "rtu_tip": "Underline key headings and provide point-wise structured explanations."
        })
        
        # Sample questions from this subject
        sampled_qs = random.sample(qlist, min(18, len(qlist)))
        for q in sampled_qs:
            q_text = q["text"]
            q_marks = q["marks"]
            q_year = q["year"]
            q_part = q["part"]
            s_name = q["subject_name"]
            
            # Bilingual toggle
            is_hinglish = random.random() < 0.5
            
            if q_marks >= 8 or "C" in q_part or "B" in q_part:
                # 10-Mark / 15-Mark comprehensive answer
                if is_hinglish:
                    user_p = f"{subj_code} exam ke liye yeh question samjhao 10 marks ke format mein: {q_text}"
                    asst_r = (
                        f"### RTU Senior Tutor: 10-Mark Model Answer ({subj_code} - {q_part})\n\n"
                        f"**Past Exam Reference**: RTU {q_year} Examination ({s_name})\n\n"
                        f"#### 1. Core Definition & Academic Concept (2 Marks)\n"
                        f"Yeh topic **{s_name}** ka fundamental concept hai. Iska primary objective system ki efficiency, reliability aur accuracy maintain karna hai. "
                        f"Question requirement: \"{q_text[:100]}...\". Isme core theoretical definition clearly mention karke examiner ko conceptual clarity show karni hai.\n\n"
                        f"#### 2. Examiner-Oriented Diagram Guide (3 Marks)\n"
                        f"- Answer sheet ke center mein neat diagram banayein aur box karein.\n"
                        f"- **Diagram Structure**: {spec['diagram']}\n"
                        f"- Har component ko clear block name aur arrows se input/output flow indicate karein.\n\n"
                        f"#### 3. Working Procedure & Step-by-Step Mechanism (3 Marks)\n"
                        f"1. **Initialization Step**: System parameters, inputs aur state variables ko define karein.\n"
                        f"2. **Processing Logic**: Core algorithmic logic ya theoretical principles execute hote hain.\n"
                        f"3. **State Transitions / Output**: Desired result calculate hota hai aur integrity constraints verify hoti hain.\n"
                        f"4. **Technical Formulation**: {spec['formula']}\n\n"
                        f"#### 4. Concrete Example / Numerical Trace (1 Mark)\n"
                        f"- Ek practical input set le kar trace karein: Example ke through examiner ko working prove karke dikhana scoring hota hai.\n\n"
                        f"#### 5. RTU Senior Scoring Tip\n"
                        f"- **Examiner Expectation**: {spec['rtu_tip']}\n"
                        f"- Part C mein points ko paragraph ke badle bullet points aur numbered steps mein likhein!"
                    )
                else:
                    user_p = f"Explain this {subj_code} RTU examination question for 10 marks: {q_text}"
                    asst_r = (
                        f"### RTU Senior Tutor: 10-Mark Standard Answer Guide ({subj_code} - {q_part})\n\n"
                        f"**Exam Grounding**: RTU {q_year} Exam Paper ({s_name})\n\n"
                        f"#### 1. Fundamental Definition & Scope (2 Marks)\n"
                        f"This topic is an essential foundation of **{s_name}**. Addressing the query: \"{q_text[:100]}...\". "
                        f"The formal definition establishes the underlying architectural principles and performance objectives required under the RTU syllabus.\n\n"
                        f"#### 2. Architectural Diagram / Schematic Guide (3 Marks)\n"
                        f"- **Visual Guide**: {spec['diagram']}\n"
                        f"- Always enclose diagrams inside a dedicated border with labeled data flows and interface connections.\n\n"
                        f"#### 3. Step-by-Step Technical Working & Mechanics (3 Marks)\n"
                        f"1. **Stage 1 (Setup & Preconditions)**: Initialize required data structures, registers, or state constraints.\n"
                        f"2. **Stage 2 (Core Execution)**: Process input vectors according to the governing algorithm or theorem.\n"
                        f"3. **Stage 3 (Verification & Output)**: Ensure boundary conditions and invariant properties are strictly satisfied.\n"
                        f"4. **Governing Formula / Mechanics**: {spec['formula']}\n\n"
                        f"#### 4. Practical Trace / Asymptotic Complexity (1 Mark)\n"
                        f"- Walk through a concrete numerical case or execution trace to demonstrate practical validity to the examiner.\n\n"
                        f"#### 5. RTU Examiner Scoring Strategy\n"
                        f"- **Key Tip**: {spec['rtu_tip']}\n"
                        f"- Highlight key terminology and write final conclusions inside prominent highlight boxes."
                    )
            else:
                # 2-Mark Crisp Part A answer
                if is_hinglish:
                    user_p = f"{subj_code} Part A ke 2-mark ke liye iska concise answer do: {q_text}"
                    asst_r = (
                        f"### RTU Senior Tutor: 2-Mark Crisp Answer (Part A)\n\n"
                        f"**Core Definition**:\n"
                        f"{q_text[:120]}...\n\n"
                        f"**Key Points & Formula**:\n"
                        f"- {spec['formula']}\n"
                        f"- Standard technical terminology aur RTU CBCS syllabus criteria ke mutabik direct point-to-point statement likhein.\n\n"
                        f"*Exam Tip: Part A mein 2-3 lines se zyada time waste mat karo, direct definition aur standard formula likh kar aage badho!*"
                    )
                else:
                    user_p = f"Provide a concise 2-mark answer for Part A in {subj_code}: {q_text}"
                    asst_r = (
                        f"### RTU Senior Tutor: 2-Mark Crisp Answer (Part A)\n\n"
                        f"**Definition & Exact Statement**:\n"
                        f"Regarding \"{q_text[:120]}...\": This is defined as the direct property ensuring functional correctness in {s_name}.\n\n"
                        f"**Crucial Formula / Rule**:\n"
                        f"- {spec['formula']}\n\n"
                        f"*Scoring Tip: State the formal definition in 2 to 3 sentences and write the governing equation to secure full 2/2 marks.*"
                    )

            data.append({
                "task": "explain_topic",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_p},
                    {"role": "assistant", "content": asst_r}
                ]
            })
            explain_count += 1

    print(f"Synthesized {explain_count} authentic explain_topic training examples")

    # ==========================================================================
    # TASK 2: quiz_generation (5-Question Practice Drills with Citations)
    # ==========================================================================
    quiz_count = 0
    for subj_code, qlist in subj_questions.items():
        if len(qlist) < 5:
            continue
        for unit_i in range(1, 4):
            sampled_quiz_qs = random.sample(qlist, min(5, len(qlist)))
            is_hinglish = random.random() < 0.5
            s_name = qlist[0]["subject_name"]
            
            if is_hinglish:
                user_p = f"{subj_code} Unit {unit_i} ({s_name}) ke liye ek 5-question exam drill quiz banao."
                asst_r = (
                    f"### RTU Senior Tutor Practice Quiz: {subj_code} Unit {unit_i} ({s_name})\n\n"
                    f"Yeh 5 authentic RTU exam-style questions hain jo aapki concept clarity aur paper timing test karenge:\n\n"
                    f"**Q1. (Part A - 2 Marks)**: {sampled_quiz_qs[0]['text'][:90]}...?\n"
                    f"- *Answer Key*: Direct 2-sentence formal definition with standard technical notation.\n\n"
                    f"**Q2. (Part A - 2 Marks)**: {sampled_quiz_qs[1]['text'][:90]}...?\n"
                    f"- *Answer Key*: State governing properties and write time/space complexity or truth table.\n\n"
                    f"**Q3. (Part B - 8 Marks)**: {sampled_quiz_qs[2]['text'][:110]}...?\n"
                    f"- *Answer Key*: Draw neat architectural diagram, explain step-by-step mechanism, and trace one sample input.\n\n"
                    f"**Q4. (Part B - 8 Marks)**: {sampled_quiz_qs[3]['text'][:110]}...?\n"
                    f"- *Answer Key*: Contrast mechanisms point-by-point in a tabular comparison format.\n\n"
                    f"**Q5. (Part C - 15 Marks)**: {sampled_quiz_qs[4]['text'][:130]}...?\n"
                    f"- *Answer Key*: Complete mathematical derivation / algorithmic trace with time complexity analysis and RTU examiner scoring tips.\n\n"
                    f"**Self-Evaluation Rubric**: 25+ Marks / 35 = Ready for RTU Semester Exam!"
                )
            else:
                user_p = f"Generate a 5-question examination practice drill for {subj_code} Unit {unit_i} ({s_name})."
                asst_r = (
                    f"### RTU Senior Tutor Practice Quiz: {subj_code} Unit {unit_i} ({s_name})\n\n"
                    f"Here is an examination-oriented practice set calibrated to the RTU CBCS marking distribution:\n\n"
                    f"**Question 1 [Part A - 2 Marks]**: {sampled_quiz_qs[0]['text'][:90]}...?\n"
                    f"- *Solution Guide*: 2-3 sentence definition citing standard nomenclature.\n\n"
                    f"**Question 2 [Part A - 2 Marks]**: {sampled_quiz_qs[1]['text'][:90]}...?\n"
                    f"- *Solution Guide*: State core constraints and governing formula.\n\n"
                    f"**Question 3 [Part B - 8 Marks]**: {sampled_quiz_qs[2]['text'][:110]}...?\n"
                    f"- *Solution Guide*: Schematic block diagram + step-by-step procedure + numerical validation.\n\n"
                    f"**Question 4 [Part B - 8 Marks]**: {sampled_quiz_qs[3]['text'][:110]}...?\n"
                    f"- *Solution Guide*: Structural comparison table with at least 5 distinct comparative parameters.\n\n"
                    f"**Question 5 [Part C - 15 Marks]**: {sampled_quiz_qs[4]['text'][:130]}...?\n"
                    f"- *Solution Guide*: Full comprehensive proof or algorithmic simulation with asymptotic recurrence analysis.\n\n"
                    f"**Benchmark**: Aim to solve this set in under 45 minutes to replicate exam conditions."
                )

            data.append({
                "task": "quiz_generation",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_p},
                    {"role": "assistant", "content": asst_r}
                ]
            })
            quiz_count += 1

    print(f"Synthesized {quiz_count} quiz_generation training examples")

    # ==========================================================================
    # TASK 3: answer_grading (0-10 Rubric Breakdown + Senior Feedback)
    # ==========================================================================
    grading_count = 0
    grading_scenarios = [
        ("DSA", "QuickSort Algorithm", "QuickSort divides array using pivot element. Code: void quicksort(int a[], int l, int r) { int p = partition(a, l, r); quicksort(a, l, p-1); quicksort(a, p+1, r); } Best case O(n log n).", 6),
        ("DBMS", "Normalization 2NF and 3NF", "2NF means no partial dependency. All non-key attributes depend on primary key. 3NF means no transitive dependency. If A->B and B->C then transitive.", 7),
        ("TOC", "DFA construction for strings starting with 'ab'", "DFA has states q0, q1, q2. From q0 on 'a' goes to q1. From q1 on 'b' goes to q2. q2 is final state. Dead state for wrong symbols.", 8),
        ("OS", "Round Robin Scheduling", "Round Robin gives equal time quantum to all processes in circular queue. If quantum is 2ms, P1 runs for 2ms then P2 runs. Context switching happens.", 6),
        ("DCCN", "CSMA/CD protocol", "Carrier Sense Multiple Access with Collision Detection. Station listens before transmit. If collision occurs, jamming signal is sent and stations wait random backoff time.", 7),
        ("AI", "A* Search Algorithm", "A* uses f(n) = g(n) + h(n). g(n) is actual cost and h(n) is heuristic. It expands node with minimum f(n). It is complete and optimal if h(n) is admissible.", 8),
        ("MI", "8085 Accumulator and Flags", "Accumulator is 8-bit register A. Flags are Sign, Zero, Auxiliary Carry, Parity, Carry. They are set after ALU operations.", 6),
        ("DE", "Full Adder using two Half Adders", "Full adder adds 3 bits. Sum = A xor B xor Cin. Carry = AB + Cin(A xor B). First half adder adds A and B, second adds sum and Cin.", 8),
        ("RTS", "Rate Monotonic Scheduling", "RMS is static priority preemptive scheduling. Shorter period means higher priority. Utilization bound is n*(2^(1/n)-1). If U <= 69% tasks are schedulable.", 8),
        ("DS", "Lamport Logical Clocks", "Lamport clocks order events in distributed systems without physical clocks. Rule 1: increment before event. Rule 2: message contains timestamp, receiver sets clock = max(local, msg) + 1.", 8),
    ]

    for subj, topic, student_ans, score in grading_scenarios:
        # English grading
        user_p_en = f"Please grade my 10-mark exam answer for {subj} on '{topic}':\n\n\"{student_ans}\""
        asst_r_en = (
            f"### RTU Senior Tutor: Answer Evaluation ({score}/10 Marks)\n\n"
            f"**Subject**: {subj} | **Topic**: {topic}\n\n"
            f"#### Rubric Score Breakdown:\n"
            f"- **1. Definition & Core Concept**: {min(2, score // 3 + 1)}/2 Marks (Conceptually sound, accurately identifies fundamental principles)\n"
            f"- **2. Architectural Diagram / Schematic**: {max(1, score // 3)}/3 Marks ({'Missing visual layout; drawing a labeled box diagram is compulsory in RTU!' if score < 8 else 'Clear logic flow; neatly framed'}) \n"
            f"- **3. Procedural Steps & Working**: {min(3, score // 3 + 1)}/3 Marks (Correct algorithmic progression, covers essential operations)\n"
            f"- **4. Complexity / RTU Scoring Conventions**: {max(1, score - 6)}/2 Marks (Stated asymptotic bounds or key identities)\n\n"
            f"#### Senior Feedback & Examiner Insight:\n"
            f"- **What you did well**: Direct, no-fluff explanation that shows you understand the core mechanism.\n"
            f"- **Where you lost marks**: RTU examiners specifically award 3 marks for labeled diagrams and trace tables. Without a visual diagram, you lose marks immediately in Part B and Part C.\n"
            f"- **How to get 10/10**: Draw the standard diagram inside a pencil border, write the formal recurrence relation, and state the Best/Average/Worst case complexities explicitly in a summary table!"
        )
        data.append({
            "task": "answer_grading",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p_en},
                {"role": "assistant", "content": asst_r_en}
            ]
        })

        # Hinglish grading
        user_p_hi = f"Bhaiya maine {subj} ke '{topic}' ke liye yeh answer likha hai RTU exam mein. Check karke marks batao:\n\n\"{student_ans}\""
        asst_r_hi = (
            f"### RTU Senior Tutor: Answer Grading ({score}/10 Marks)\n\n"
            f"**Subject**: {subj} | **Topic**: {topic}\n\n"
            f"#### Score Breakdown (RTU Rubric):\n"
            f"- **1. Definition & Terminology**: {min(2, score // 3 + 1)}/2 Marks (Basic concept sahi likha hai)\n"
            f"- **2. Diagram & Layout**: {max(1, score // 3)}/3 Marks ({'Diagram missing hai! RTU mein bina diagram ke marks katte hain' if score < 8 else 'Diagram guide achhi hai'})\n"
            f"- **3. Working Steps & Algorithm**: {min(3, score // 3 + 1)}/3 Marks (Steps sahi order mein hain)\n"
            f"- **4. Complexity / Example**: {max(1, score - 6)}/2 Marks (Example aur time complexity covered hai)\n\n"
            f"#### Senior Advice for RTU Examiner:\n"
            f"- **Achi baat**: Aapne seedhe point ki baat likhi hai aur concept clear hai.\n"
            f"- **Kahan marks kate**: RTU examiner Part B aur Part C mein sheet bharne aur neat diagram banane ke marks deta hai. Diagram ko answer sheet ke center mein box bana ke draw karo.\n"
            f"- **10/10 lane ka tarika**: Heading ko black pen se underline karo, time complexity $O(n \\log n)$ ko box mein band karo, aur 1 chota input trace karke dikhao!"
        )
        data.append({
            "task": "answer_grading",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p_hi},
                {"role": "assistant", "content": asst_r_hi}
            ]
        })
        grading_count += 2

    print(f"Synthesized {grading_count} answer_grading training examples")

    # ==========================================================================
    # TASK 4: structured_extraction (Raw OCR Question Paper to Structured JSON)
    # ==========================================================================
    extraction_count = 0
    for subj_code, qlist in subj_questions.items():
        sampled_extract = random.sample(qlist, min(3, len(qlist)))
        for q in sampled_extract:
            ocr_noise = f"Roll No............. [Total No. of Pages : 2]\nB.Tech. {q['subject_code']} Examination - {q['year']}\n{q['part']}\nQ.{random.randint(1, 8)} {q['text']} [{q['marks']}]"
            extracted_json = {
                "subject": subj_code,
                "subject_name": q["subject_name"],
                "year": q["year"],
                "part": q["part"],
                "marks": q["marks"],
                "question_text": q["text"],
                "unit": q["unit_no"]
            }

            user_p = f"Extract the structured metadata and question information from this raw RTU OCR paper text into JSON:\n\n\"\"\"\n{ocr_noise}\n\"\"\""
            asst_r = f"```json\n{json.dumps(extracted_json, indent=2)}\n```"

            data.append({
                "task": "structured_extraction",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_p},
                    {"role": "assistant", "content": asst_r}
                ]
            })
            extraction_count += 1

    print(f"Synthesized {extraction_count} structured_extraction training examples")

    # ==========================================================================
    # TASK 5: refusal_idk (Zero Hallucination Guardrails for Out-of-Syllabus)
    # ==========================================================================
    refusal_queries = [
        ("How to bake chocolate chip cookies?", False),
        ("Who won the 2023 Cricket World Cup?", False),
        ("Explain quantum gravity and string theory.", False),
        ("What is the latest stock price of Apple and Tesla?", False),
        ("Write a script to hack a Facebook account.", False),
        ("Recommend the best romantic movies on Netflix.", False),
        ("How to change engine oil in a Honda Civic?", False),
        ("Tell me the plot summary of Harry Potter.", False),
        ("Can you help me solve my medical symptoms for fever?", False),
        ("How do I trade Bitcoin and make quick money?", False),
        ("Biryani banane ki recipe batao bhai.", True),
        ("Bollywood ka sabse famous actor kaun hai?", True),
        ("Instagram followers badhane ka tarika batao.", True),
        ("Wifi ka password hack kaise karein?", True),
        ("General Knowledge ke 10 questions poochho.", True),
        ("Weather kaisa rahega Jaipur mein kal?", True),
        ("Share market mein invest kaise karein?", True),
        ("Pubg / BGMI mein pro player kaise bane?", True),
        ("Car driving kaise sikhein?", True),
        ("Kavita likh kar do prem par.", True)
    ]

    refusal_count = 0
    for query, is_hinglish in refusal_queries:
        if is_hinglish:
            user_p = query
            asst_r = (
                "Main RTU Senior Academic Tutor hoon aur strictly Rajasthan Technical University (RTU) ke engineering syllabus "
                "aur B.Tech semester exams ki preparation ke liye trained hoon. Yeh question RTU syllabus ka part nahi hai, "
                "isliye main iska answer provide nahi kar sakta. "
                "Aap mujhse apne engineering subjects (jaise DSA, DBMS, TOC, DCCN, OS, DAA, AI, ML, DE, MI, TC, MEFA) ke "
                "concepts, algorithms, 10-mark model answers, ya past year papers pooch sakte hain!"
            )
        else:
            user_p = query
            asst_r = (
                "I am the RTU Senior Academic Assistant, specialized strictly in Rajasthan Technical University (RTU) "
                "B.Tech engineering curricula and past examination preparation. This query is outside the scope of the "
                "RTU engineering syllabus, and I cannot assist with non-academic or general trivia requests. "
                "Please feel free to ask questions related to your RTU subjects (such as DSA, DBMS, DCCN, TOC, OS, DAA, AI, DE, MI, AEM) "
                "and I will be happy to provide standard 10-mark structures, diagrams, and past year question analysis!"
            )

        data.append({
            "task": "refusal_idk",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_p},
                {"role": "assistant", "content": asst_r}
            ]
        })
        refusal_count += 1

    print(f"Synthesized {refusal_count} refusal_idk training examples")

    # Shuffle dataset
    random.shuffle(data)

    # Write output JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for ex in data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print(f"Expanded Dataset Successfully Created: {OUTPUT_FILE}")
    print(f"Total Training Examples: {len(data)}")
    print("=" * 60)

    # Print Task Distribution
    task_dist = {}
    for ex in data:
        t = ex.get("task", "unknown")
        task_dist[t] = task_dist.get(t, 0) + 1
    for t, cnt in sorted(task_dist.items()):
        print(f"  - {t:<25}: {cnt:>4} examples ({cnt/len(data)*100:.1f}%)")

if __name__ == "__main__":
    build_dataset()
