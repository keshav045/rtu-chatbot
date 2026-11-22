# RTU AI Academic Assistant: The Senior Tutor Exam Coach

> An offline-first, past-paper-grounded AI academic coach built specifically for Rajasthan Technical University (RTU) B.Tech engineering students.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0468D7.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Keyword%20Search-BM25Okapi-FF6F00.svg)](https://github.com/dorianbrown/rank_bm25)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama%20(rtu--senior)-black.svg)](https://ollama.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange.svg)](https://smith.langchain.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why We Built This

Preparing for RTU semester exams is notoriously stressful. Syllabus documents are long, question paper archives are disorganized, and generic AI tools do not understand how RTU evaluators grade answers. 

This project transforms 724 RTU examination papers (2010 to 2024) and official syllabus archives into a specialized, offline-first exam assistant that behaves like an empathetic senior tutor.

---

## What It Does

1. **Senior Tutor Exam Format**: Provides complete 10-mark and 2-mark answer blueprints (Technical Definition, Examiner Diagram Guide, Working Steps, Numerical/Trace Example, and Examiner Scoring Tips).
2. **Bilingual Conversational Fluency**: Answers in clear technical English or natural Hindi/Hinglish.
3. **Verifiable Past Paper Citations**: Every answer links back to authentic RTU papers with subject code, exam year, section, marks, and physical question text.
4. **Empirical Topic Weightage**: Data-backed frequency analysis identifying high-yield topics across 724 papers.
5. **Crash Revision Planner**: Dynamic hour-by-hour study roadmap based on the time you have left before an exam.
6. **Answer Grading with 0-10 Rubric**: Evaluates practice answers against university grading standards with constructive feedback.
7. **Out-of-Syllabus Guardrail**: Politely refuses non-academic queries to maintain focus on engineering preparation.

---

## Dataset & Technical Coverage

* **724 Past Papers on Disk** (2010 to 2024).
* **150 Canonical Papers Extracted** using PyMuPDF + RapidOCR.
* **1,534 Verified Exam Questions** cataloged with complete metadata in SQLite (`rtu_metadata.db`).
* **180 Syllabus Topics across 109 Units** for 22 core subjects (Semesters 3 through 8).
* **Hybrid Search Engine**: FAISS IndexFlatIP (`BAAI/bge-small-en-v1.5`) + BM25Okapi + Reciprocal Rank Fusion ($k=60$) + Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`).
* **Custom Fine-Tuned Model (`rtu-senior`)**: Fine-tuned Qwen2.5-1.5B via QLoRA on a 748-pair curated RTU dataset, quantized to GGUF, and running locally in Ollama with `num_ctx = 4096`.
* **Production Observability**: Integrated LangSmith tracing for monitoring latency, token usage, and retrieval execution.

---

## Quick Start (Backend)

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server**:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

3. **Check API Health**:
   Open `http://127.0.0.1:8000/docs` to test interactive Swagger documentation.

---

## Running Automated Tests

Run the full automated test suite:
```bash
pytest tests -o pythonpath="."
```
All 28 tests pass with 100% success across ingestion, hybrid retrieval, endpoints, and security guardrails.
