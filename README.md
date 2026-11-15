---
title: RTU AI Academic Assistant
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# RTU AI Academic Assistant
> An AI-powered exam coach and past-paper intelligence platform designed specifically for Rajasthan Technical University (RTU) B.Tech students.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20ZeroGPU-blue.svg)](https://huggingface.co/spaces/keshavagarwal779/rtu-exam-assistant)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0468D7.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Keyword%20Search-BM25Okapi-FF6F00.svg)](https://github.com/dorianbrown/rank_bm25)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Live Deployment

The system is deployed live on Hugging Face Spaces backed by dynamic **NVIDIA A10G ZeroGPU** hardware:
* **Interactive Live App**: [https://huggingface.co/spaces/keshavagarwal779/rtu-exam-assistant](https://huggingface.co/spaces/keshavagarwal779/rtu-exam-assistant)
* **Direct Webview**: [https://keshavagarwal779-rtu-exam-assistant.hf.space](https://keshavagarwal779-rtu-exam-assistant.hf.space)

---

## Overview

Preparing for university engineering exams usually requires searching through dozens of past-year question papers (PYQs), estimating repeated question frequency, and aligning responses with university scoring expectations.

**RTU AI Academic Assistant** converts 14 years of RTU past question papers and syllabus archives into a verified academic study companion:
1. **Senior Tutor Exam Assistant**: Natural English and Hinglish conversational tutor providing exam-oriented responses (Definitions, Block Diagrams, Algorithmic Steps, Working Examples, and RTU Scoring Tips).
2. **Verifiable Paper Citations**: Every answer links directly to authentic RTU question papers with subject code, year, section, marks, and physical PDF page numbers.
3. **High-Yield Topic Rankings**: Empirical question frequency analysis identifying high-weightage topics across past examinations.
4. **Crash Course Revision Planner**: Structured hour-by-hour revision schedules calculated from available preparation time.
5. **Interactive Practice & Evaluation**: Practice questions with scoring rubrics aligned to university grading criteria.
6. **Past Papers Archive**: Searchable database of past exam questions and university papers.

---

## Benchmark Performance

Evaluated on 100 test queries across core engineering subjects:

* **Hit@1**: **89.4%** (top retrieved question is directly relevant)
* **Hit@5**: **95.3%** (relevant exam question captured in top 5)
* **Hit@10**: **97.7%** (captured in top 10)
* **MRR (Mean Reciprocal Rank)**: **0.922**
* **Recall@5**: **87.7%**
* **Out-of-Scope Guardrail ROC-AUC**: **0.9702**
* **Pass Rate**: **28/28 Automated Pytest Integration, Retrieval & Security Tests (100%)**

---

## Architecture

The system utilizes a 4-stage hybrid retrieval and neural reranking pipeline:

```
User Query (English / Hinglish)
           │
           ├───> Dense Retrieval: FAISS IndexFlatIP (BAAI/bge-small-en-v1.5) [k=25]
           │
           └───> Sparse Retrieval: BM25Okapi Tokenized Lexical Matching [k=25]
                         │
                         ▼
           Reciprocal Rank Fusion (RRF, k=60)
                         │
                         ▼
           Neural Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
                         │
                         ▼
           Out-of-Scope Score Gate & Context Extraction
                         │
                         ▼
           LLM Synthesis (Groq / Gemini / Local) + Verifiable Citations
```

---

## Core Engineering Subjects Covered

* **DSA**: Data Structures and Algorithms (`3CS4-05` / `3E1202`)
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

## Local Execution

### 1-Click Unified Launcher (Windows)
Double-click:
```cmd
run_app.bat
```
This launches the FastAPI backend on `http://127.0.0.1:8000` and the Streamlit interface on `http://localhost:8501`.

### Gradio Interface
Run directly:
```bash
python app.py
```
This launches the Gradio web application locally on `http://127.0.0.1:7860`.

---

## Automated Test Suite

To run all 28 automated integration, retrieval, security, and regression tests:
```bash
pytest rtu-ai-assistant/backend/tests -o pythonpath="rtu-ai-assistant/backend"
```
Test results:
* Phase 1 (Ingestion & SQLite schema): 3/3 passed
* Phase 2 (Dense & Sparse Retrieval, RRF): 6/6 passed
* Phase 3 (FastAPI Endpoints & Citations): 6/6 passed
* Phase 4 (v2 Features & Cross-Encoder): 6/6 passed
* Phase 5 (Security Guardrails, Anti-Jailbreak, Anti-Leakage & Feedback): 7/7 passed
