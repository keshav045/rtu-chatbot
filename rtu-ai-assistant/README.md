# RTU AI Academic Assistant
> An AI-powered exam coach and past-paper intelligence platform designed specifically for Rajasthan Technical University (RTU) B.Tech students.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0468D7.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Keyword%20Search-BM25Okapi-FF6F00.svg)](https://github.com/dorianbrown/rank_bm25)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What is this project?

Preparing for university engineering exams usually means digging through dozens of past-year question papers (PYQs), guessing which topics are actually repeated, and trying to figure out how RTU examiners score answers.

**RTU AI Academic Assistant** solves this by turning 14 years of RTU past question papers and official syllabus documents into an interactive study companion. Instead of reading flat PDFs, students can:
1. Ask questions in natural English or Hinglish and get complete, exam-standard answers (Definitions, Diagrams, Working Steps, Examples, and RTU Scoring Tips).
2. See exactly which paper, year, section, and page number every answer is sourced from.
3. Generate customized, hour-by-hour crash revision roadmaps based on how many hours they have left before an exam.
4. View data-backed topic rankings showing which topics carry the highest mark weightage across past papers.
5. Practice real past exam questions and get instant feedback scored on RTU's 0–10 marks rubric.

---

## Key Features

### 1. Senior Tutor Exam Chatbot
* **Exam-Oriented Structure**: Answers follow the standard RTU 10-mark and 2-mark formats expected by examiners:
  * Core Technical Definition (2–3 sentences)
  * Neat Block Diagram / Trace Table description
  * Step-by-step Algorithm or Procedural Derivation
  * Concrete Worked Example
  * Senior Exam Tip (highlighting common deduction traps)
* **Bilingual Support**: Understands both English and conversational Hinglish (e.g., *"Iska worst case time complexity kya hota hai?"*).
* **Zero-Hallucination Citations**: Every answer links back to verified RTU question papers with subject code, year, section, marks, and physical PDF page numbers.
* **Out-of-Syllabus Guardrail**: Automatically detects and refuses non-engineering queries (e.g., cricket, politics, or competitive exam cutoffs) with an ROC-AUC of 0.97.

### 2. High-Yield Topic Weightage
* Analyzes past examination papers to empirically calculate:
  * Total marks yielded by each topic
  * Number of distinct exam years the topic appeared
  * Frequency of long-answer ($\ge 8$ marks) vs. short-answer ($\le 2$ marks) questions
* Honestly tags topics as `HIGH PRIORITY (Frequently Asked)` or `HIGH WEIGHTAGE (Core Exam Topic)`.

### 3. Crash Course Study Planner
* Dynamically generates an hour-by-hour study schedule based on available preparation time (from 30 minutes up to 12 hours).
* Prioritizes high-weightage topics first using a knapsack-style time budget, automatically reserving time for a comprehensive rapid formula and diagram revision block before the exam.

### 4. Practice Quiz & Evaluator
* Serves real past questions for the selected subject.
* Evaluates student answers on a strict 0–10 rubric:
  * Definition (0–2.5 marks)
  * Diagram & Architecture (0–2.5 marks)
  * Procedural Steps (0–3.0 marks)
  * Example / Derivation (0–2.0 marks)
* Detects empty or off-topic answers (e.g. random sentences score $\le 0.5$, explicit refusals score $0.0$) and offers actionable examiner feedback.

### 5. Past Papers Archive & Scanned PDF Viewer
* Filter and browse original RTU question papers by subject, semester, and year.
* Built-in viewer opens original scanned examination PDFs with direct page jumping.

---

## System Architecture

```
[ Student / Browser ]
         │
         ▼
[ Frontend: HTML5 + CSS3 + Vanilla JavaScript ]
  ├── Chat Interface & Quick Prompts
  ├── Study Setup Modal (Year, Branch, Subject, Hours, Mode)
  ├── 10-Mark / 2-Mark Topic Weightage Explorer
  ├── Hour-by-Hour Crash Course Planner
  ├── Interactive Practice Quiz with 0-10 Rubric Breakdown
  └── Scanned Question Paper Archive & PDF Streamer
         │
         │ REST API (Port 8000 /api)
         ▼
[ FastAPI Backend ]
  ├── Intent Router (Greeting, Study Plan, Topic Importance, PYQ, Question)
  ├── Query Rewriter (Resolves conversational follow-ups & pronouns)
  ├── SQLite Database (`rtu_metadata.db`)
  │     ├── 123 Examination Papers (2012–2026)
  │     ├── 387 Canonical Verified Exam Questions
  │     └── 154 Curriculum Units & Topics (RTU 2018 CBCS Scheme)
  │
  ├── Unified Hybrid Retrieval Pipeline
  │     ├── Dense Vector Retrieval: BAAI/bge-small-en-v1.5 (FAISS Cosine Search)
  │     ├── Sparse Keyword Retrieval: BM25Okapi (Exact Subject Terms & Codes)
  │     ├── Reciprocal Rank Fusion (RRF, k=60) + Metadata Filtering
  │     └── Cross-Encoder Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
  │
  └── LLM Provider Layer
        ├── Local Ollama (`qwen2.5:1.5b` or fine-tuned `rtu-senior`)
        ├── Groq Cloud (`llama-3.3-70b-versatile`)
        ├── Google Gemini (`gemini-1.5-flash`)
        └── Offline Academic Mentor (100% deterministic fallback)
```

---

## Benchmark & Retrieval Performance

The retrieval engine was evaluated using an automated benchmark suite of 100 queries across all core engineering subjects:

| Metric | Score | Interpretation |
|---|---|---|
| **Hit@1** | **89.4%** | In ~90% of queries, the #1 retrieved document is directly relevant |
| **Hit@5** | **95.3%** | 95.3% of queries contain the relevant RTU past question in top 5 |
| **Hit@10** | **97.7%** | Captures almost 98% of relevant questions in top 10 candidates |
| **MRR (Mean Reciprocal Rank)** | **0.922** | Relevant documents almost always rank at position 1 or 2 |
| **Recall@5** | **87.7%** | Retrieves ~88% of all relevant questions across the entire dataset |
| **Guardrail ROC-AUC** | **0.9702** | Excellent separation between in-syllabus and out-of-syllabus queries |
| **Latency (p50 / median)** | **1.09 s** | Runs fast on standard consumer CPU without needing a dedicated GPU |

---

## Supported Core Subjects

All subjects are mapped to the official RTU B.Tech scheme:

* **DSA**: Data Structures and Algorithms (`3CS4-05`)
* **DBMS**: Database Management System (`4CS4-05`)
* **DCCN**: Data Communication and Computer Networks (`4CS4-07`)
* **OOP**: Object Oriented Programming in C++ (`3CS4-06`)
* **DE**: Digital Electronics (`3CS3-04`)
* **AEM**: Advanced Engineering Mathematics (`3CS2-01`)
* **TOC**: Theory of Computation (`4CS4-06`)
* **SE**: Software Engineering (`3CS4-07`)
* **DMS**: Discrete Mathematics Structure (`4CS2-01`)
* **MEFA**: Managerial Economics and Financial Accounting (`3CS1-03`)
* **MI**: Microprocessor and Interfaces (`5CS3-01`)
* **TC**: Technical Communication (`4CS1-02`)

---

## Project Structure

```
rtu project/
├── cleaned_dataset/              # Scanned PDFs and extracted question OCR data
├── run_app.bat                   # 1-Click launcher for Windows
└── rtu-ai-assistant/
    ├── Dockerfile                # Production backend container build
    ├── docker-compose.yml        # Multi-container orchestration (Backend + Frontend)
    ├── README.md                 # Project documentation
    ├── run_backend.bat           # Run FastAPI server directly
    ├── run_frontend.bat          # Open browser interface
    ├── start_all.bat             # Unified launcher
    │
    ├── backend/
    │   ├── requirements.txt      # Python dependencies
    │   ├── app/
    │   │   ├── main.py           # FastAPI entrypoint & static mounts
    │   │   ├── core/config.py    # Paths, thresholds & settings
    │   │   ├── database/         # SQLite models & database session
    │   │   ├── retrieval/        # FAISS, BM25, Hybrid RRF & Reranker
    │   │   ├── services/         # Chat, Intent, Planner, Quiz, Importance
    │   │   └── llm/              # Ollama, Groq, Gemini & Offline providers
    │   ├── data/
    │   │   ├── storage/          # rtu_metadata.db (SQLite database)
    │   │   └── indexes/          # FAISS index and BM25 pickle
    │   └── tests/                # Automated verification & test suites
    │
    ├── frontend/
    │   ├── index.html            # Single-page academic dashboard
    │   ├── style.css             # Dark-mode responsive styling
    │   ├── app.js                # Frontend state, API client & UI rendering
    │   └── Dockerfile            # NGINX Alpine static web server
    │
    ├── evaluation/               # Benchmark scripts & test datasets
    └── models/                   # Local GGUF models & Modelfile
```

---

## How to Run the Project

### Prerequisites
* Python 3.10 or 3.11 installed
* Windows 10/11 (or Linux/macOS)
* *(Optional)* [Ollama](https://ollama.com/) if you wish to run local LLMs (e.g. `qwen2.5:1.5b`)

### Method 1: One-Click Launch (Windows)
Simply double-click:
```cmd
run_app.bat
```
This will automatically launch the backend server on `http://127.0.0.1:8000` and open the app in your default web browser.

---

### Method 2: Manual Terminal Launch

1. **Install dependencies**:
   ```bash
   cd rtu-ai-assistant/backend
   pip install -r requirements.txt
   ```

2. **Start the FastAPI backend server**:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

3. **Open the web dashboard**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:8000/
   ```

---

### Method 3: Docker Deployment

Run both backend and frontend via Docker Compose:
```bash
cd rtu-ai-assistant
docker-compose up --build
```
* **Frontend**: `http://localhost:5173`
* **Backend API**: `http://localhost:8000/api`
* **Interactive API Docs (Swagger)**: `http://localhost:8000/docs`

---

## Verifying the Test Suite

To run the automated verification suite:
```bash
cd rtu-ai-assistant
python backend/tests/verify_all_fixes.py
python evaluation/test_student_flow.py
```

---

## Authors & Acknowledgments
* Developed as a comprehensive final-year / capstone project for **Rajasthan Technical University (RTU)** engineering students.
* Built using open-source tools: FastAPI, FAISS, rank-bm25, sentence-transformers, and PyMuPDF.
