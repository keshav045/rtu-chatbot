---
title: RTU AI Academic Assistant
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
---

# RTU AI Academic Assistant: The Senior Tutor Exam Coach

> An offline-first, past-paper-grounded AI academic coach built specifically for Rajasthan Technical University (RTU) B.Tech engineering students.

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain%20LCEL-1C3C3C.svg)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange.svg)](https://smith.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector%20Search-FAISS-0468D7.svg)](https://github.com/facebookresearch/faiss)
[![BM25](https://img.shields.io/badge/Keyword%20Search-BM25Okapi-FF6F00.svg)](https://github.com/dorianbrown/rank_bm25)
[![Ollama](https://img.shields.io/badge/Local%20LLM-Ollama%20(rtu--senior)-black.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## The Story: Why We Built This

If you have ever prepared for an RTU semester examination, you know the drill:
* It is 2 AM before the exam, and you are digging through photocopied stacks of previous year question papers (PYQs).
* The syllabus has 5 massive units, but you only have 6 to 8 hours left to study.
* You do not know which topics are genuinely repeated, which ones carry 15 marks in Part C, and what the examiner actually expects on the answer sheet.
* Generic chatbots like ChatGPT or Gemini do not understand the RTU paper pattern: they write long essays without the block diagrams, algorithmic steps, or marks-weightage breakdowns that RTU evaluators grade on.

We built this project to solve that exact problem. Think of it as that knowledgeable 4th-year senior from your hostel who knows every past paper from the last decade, knows how marks are awarded, and explains concepts simply in English or college Hinglish with the exact structure needed to clear the exam.

---

## Live Deployments & Demos

* **Interactive Streamlit App**: `http://localhost:8501` (via local launcher)
* **Live Hugging Face Space**: [https://huggingface.co/spaces/keshavagarwal779/rtu-exam-assistant](https://huggingface.co/spaces/keshavagarwal779/rtu-exam-assistant)
* **Direct Webview**: [https://keshavagarwal779-rtu-exam-assistant.hf.space](https://keshavagarwal779-rtu-exam-assistant.hf.space)

---

## What Makes This Different

### 1. Senior Tutor Persona (English & Hinglish)
The assistant does not sound like a robotic textbook. It talks like an empathetic senior tutor. You can ask in formal English or natural Hinglish:
* *"B-tree insertion ka working trace karke samjhao 10 marks ke format mein."*
* *"TOC mein Pumping Lemma 2 marks ke liye kaise likhein?"*
* *"I have 6 hours left for DBMS exam. Give me a crash revision roadmap."*

### 2. Standard 10-Mark and 2-Mark Exam Format
Every long-form explanation is structured specifically for university evaluators:
1. **Core Technical Definition & Academic Scope (2 Marks)**: Formal statement with technical terminology.
2. **Examiner-Oriented Diagram Guide (3 Marks)**: Clear ASCII/text schematic, block layout, or state transition guide to draw on the answer sheet.
3. **Step-by-Step Working & Algorithm (3 Marks)**: Numbered steps, recurrence relations, or dataflow transitions.
4. **Concrete Trace / Numerical Example (1 Mark)**: Worked sample input proving practical validity.
5. **RTU Scoring Tip (1 Mark)**: Practical warnings about deduction traps (e.g., highlighting time complexity, underlining headings, boxing final numerical answers).

### 3. Verifiable Past Paper Citations
No hallucinations. Every answer includes citations linking directly to authentic RTU examination papers with subject code, exam year, section (Part A, B, or C), marks, and original question text.

### 4. High-Yield Topic Weightage
The system analyzes past examination papers to empirically calculate:
* Total marks yielded by each topic across past papers.
* Number of distinct exam sessions the topic appeared in.
* Frequency of Part C (15 marks) vs. Part B (8 marks) vs. Part A (2 marks) questions.
* Automatic priority tags (`HIGH PRIORITY` or `CORE EXAM TOPIC`).

### 5. Dynamic Crash Course Planner
Tell the assistant how many hours you have left (from 1 to 12 hours). It runs a knapsack-style time budget that prioritizes high-weightage topics first and reserves the final hour for formula and diagram revision.

### 6. Interactive Answer Grading (0 to 10 Rubric)
Paste your practice answer into the evaluator. It grades your answer on RTU's standard rubric:
* Definition & Academic Concept: 0 to 2.5 marks
* Diagram & Architecture: 0 to 2.5 marks
* Procedural Steps & Working: 0 to 3.0 marks
* Numerical / Example / Complexity: 0 to 2.0 marks
It tells you honestly where marks would be deducted and how to reach full marks.

### 7. Academic Screening Guardrails
Before generating answers, the system runs a multi-stage academic screener:
* **Input Scope Screener**: Blocks off-topic queries (cricket, cinema, general chatter) to preserve academic rigor.
* **Prompt Injection Defense**: Prevents instruction overriding, roleplay spoofing, and jailbreak attempts.
* **Output Sanitizer & Anti-Leakage**: Strips private tokens, system prompts, and verifies citation groundedness.

---

## Real Data & Coverage Numbers

We do not use synthetic or placeholder data. Everything is grounded in genuine university archives:

| Metric | Verified Count |
| :--- | :--- |
| **Past Examination Papers on Disk** | **724 Papers** (spanning 2010 to 2024) |
| **Canonical Extracted Question Papers** | **150 Papers** (PyMuPDF + RapidOCR high-yield engine) |
| **Searchable Exam Questions in Database** | **1,534 Questions** stored with full metadata |
| **Curriculum Topics Cataloged** | **180 Syllabus Topics** across **109 Units** |
| **Core Engineering Subjects** | **22 Active Subjects** covering Semesters 3 through 8 |
| **Dense Vector Embeddings** | **1,534 Vectors** in LangChain FAISS (`BAAI/bge-small-en-v1.5`) |
| **Sparse Lexical Index** | **1,534 Documents** in BM25Okapi |
| **Fine-Tuning Dataset** | **748 Authentic Question-Answer Pairs** |

---

## Subjects & Semesters Covered

* **Semester 3**: Data Structures and Algorithms (DSA), Object Oriented Programming in C++ (OOP), Digital Electronics (DE), Advanced Engineering Mathematics (AEM), Software Engineering (SE), Managerial Economics and Financial Accounting (MEFA).
* **Semester 4**: Database Management Systems (DBMS), Theory of Computation (TOC), Data Communication and Computer Networks (DCCN), Discrete Mathematics Structure (DMS), Technical Communication (TC).
* **Semester 5**: Operating Systems (OS), Design and Analysis of Algorithms (DAA), Microprocessor and Interfaces (MI), Computer Networks (CN).
* **Semester 6**: Artificial Intelligence (AI), Compiler Design (CD), Computer Architecture (CA), Cloud Computing (CC).
* **Semester 7**: Big Data Analytics (BDA), Machine Learning (ML), Internet of Things (IOT), Information System Security (ISS).
* **Semester 8**: Distributed Systems (DS), Digital Image Processing (DIP), Disaster Management (DM), Mobile Computing (MC), Real Time Systems (RTS).

---

## System Architecture

The architecture uses a modular LangChain LCEL pipeline with hybrid compression retrieval and neural reranking:

```
                      [ Student / User Interface ]
                      (Streamlit Web UI / Gradio)
                                   │
                                   ▼
                         [ FastAPI Backend ]
                      (Port 8000 /api endpoints)
                                   │
          ┌────────────────────────┴────────────────────────┐
          ▼                                                 ▼
[ Academic Screening Guardrails ]                [ Multi-Turn Session Memory ]
(Scope, Injection & Safety Checks)               (Maintains active subject state)
          │
          ▼
[ LangChain Hybrid Retrieval ]
   ├── Dense Vector Search: LangChain FAISS (BAAI/bge-small-en-v1.5) [k=25]
   └── Sparse Lexical Search: BM25Retriever Tokenized Match [k=25]
          │
          ▼
[ EnsembleRetriever (0.5 Dense + 0.5 Sparse) ]
          │
          ▼
[ ContextualCompressionRetriever ]
   └── CrossEncoderReranker (ms-marco-MiniLM-L-6-v2, top_n=5)
          │
          ▼
[ LCEL Senior Tutor Chain ]
   ├── PromptTemplate (10-Mark Rubric, Diagrams, Scoring Traps)
   ├── Unified ChatModel (Ollama / Groq / Gemini / OpenAI)
   └── StrOutputParser
          │
          ▼
[ LangSmith Tracing & Evaluators ] ──> Latency, Tokens, Feedback & Scores
          │
          ▼
[ Output Screening & Grounded Answer with Citations ]
```

---

## Custom Fine-Tuned Model: `rtu-senior`

Instead of relying solely on generic base models, we fine-tuned **Qwen2.5-1.5B-Instruct** using QLoRA specifically on RTU exam data.

* **Training Dataset**: 748 curated examples ([rtu_train_dataset_expanded.jsonl](file:///c:/Users/kesha/OneDrive/Desktop/rtu%20project/rtu-ai-assistant/fine_tuning/rtu_train_dataset_expanded.jsonl)) covering 10-mark structures, 2-mark brevity, quiz generation, answer grading, OCR extraction, and out-of-syllabus refusals.
* **Training Platform**: Free Google Colab T4 GPU (16 GB VRAM) using our automated notebook ([RTU_Senior_Tutor_FineTuning_Colab.ipynb](file:///c:/Users/kesha/OneDrive/Desktop/rtu%20project/rtu-ai-assistant/fine_tuning/RTU_Senior_Tutor_FineTuning_Colab.ipynb)).
* **Training Metrics**:
  * Training Loss: dropped from `3.439` to `0.7449`
  * Mean Token Accuracy: **`94.83%`** across 3 epochs (141 steps)
* **Local Quantization**: Exported to 8-bit GGUF (`rtu_senior_tutor_q8.gguf`, 1.57 GB) and 4-bit GGUF (`rtu_senior_tutor_q4.gguf`, 940 MB).
* **Hardware Friendly**: Runs locally on a standard 4 GB VRAM laptop (e.g., NVIDIA GTX 1650) with `num_ctx = 4096` in local Ollama with zero external API calls.

---

## Quick Start Guide

### Option 1: 1-Click Unified Launcher (Windows)
Double-click `run_app.bat` in the project root. This automatically starts:
1. The FastAPI backend server on `http://127.0.0.1:8000`
2. The Streamlit web application on `http://localhost:8501`

### Option 2: Manual Terminal Startup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/keshav045/rtu-chatbot.git
   cd rtu-chatbot
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-lc.txt
   ```

4. **Set up Environment Variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Add your keys in `.env`:
   ```env
   # LangSmith Tracing (Optional but recommended)
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=rtu-ai-academic-assistant

   # Cloud LLM API Keys (Optional - only needed if not running local Ollama)
   GROQ_API_KEY=your_groq_key
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key

   # Default LLM Provider (ollama, groq, gemini, openai, offline)
   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_MODEL=rtu-senior
   ```

5. **Start the FastAPI backend**:
   ```bash
   python -m uvicorn app.main:app --app-dir rtu-ai-assistant/backend --host 127.0.0.1 --port 8000
   ```

6. **Start the Streamlit user interface**:
   ```bash
   streamlit run streamlit_app.py --server.port 8501
   ```

Open your browser to **`http://localhost:8501`**.

---

## Running with Local Ollama (`rtu-senior`)

If you want to run completely offline without spending any API credits:

1. Install [Ollama](https://ollama.com/).
2. Navigate to the models directory and build the model:
   ```bash
   cd rtu-ai-assistant/models
   ollama create rtu-senior -f Modelfile
   ollama run rtu-senior
   ```
3. In your `.env` file, ensure:
   ```ini
   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_MODEL=rtu-senior
   ```
4. The entire application now runs 100% locally on your machine with zero internet required.

---

## Observability & Tracing with LangSmith

Production tracing monitors retrieval latency, token consumption, and evaluation metrics in real time.

1. Create a free account at [smith.langchain.com](https://smith.langchain.com).
2. Set your environment variables in `.env`:
   ```env
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=lsv2_pt_...
   LANGCHAIN_PROJECT=rtu-ai-academic-assistant
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   ```
3. **Check Status**: Visit `http://localhost:8000/api/observability/status` to verify real-time connectivity.
4. **Sync Benchmark Dataset**: Run `POST /api/observability/sync-dataset` to upload the 300-question RTU examination benchmark directly to your LangSmith account.
5. **Interactive Feedback**: When students give thumbs-up or thumbs-down feedback in the Streamlit UI, scores are directly synced to the corresponding trace in LangSmith Cloud.

---

## Project Structure

```text
rtu-project/
├── streamlit_app.py                   # Main Streamlit Web Application
├── app.py                             # Gradio Standalone Application
├── run_app.bat                        # 1-Click Unified Windows Launcher
├── requirements.txt                   # Core dependency manifest
├── requirements-lc.txt                # LangChain & LangSmith dependencies
├── .env.example                       # Environment configuration template
│
├── rtu-ai-assistant/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── api/endpoints.py       # REST API endpoints (Chat, Search, Quiz, Planner, Observability)
│   │   │   ├── core/config.py         # System configuration & environment loading
│   │   │   ├── core/tracer.py         # LangSmith tracing engine & feedback sync
│   │   │   ├── core/langsmith_datasets.py # 300-Question RTU benchmark dataset persistence
│   │   │   ├── database/models.py     # SQLite SQLAlchemy ORM schemas
│   │   │   ├── llm/
│   │   │   │   ├── lc_models.py       # Unified LangChain ChatModel factory
│   │   │   │   └── offline_mentor_provider.py # Template engine fallback
│   │   │   ├── retrieval/
│   │   │   │   ├── lc_retriever.py    # Hybrid Ensemble + CrossEncoder compression retriever
│   │   │   │   ├── pipeline.py        # Retrieval orchestrator
│   │   │   │   └── query_rewriter.py  # Exam query expansion
│   │   │   └── services/
│   │   │       ├── chat_service.py    # Multi-turn conversational orchestrator
│   │   │       ├── lc_chain.py        # LCEL 10-Mark Senior Tutor answer chain
│   │   │       ├── screening_service.py # Academic screening guardrails
│   │   │       ├── planner_service.py # Crash revision schedule planner
│   │   │       └── quiz_service.py    # Practice exam & evaluation generator
│   │   ├── data/
│   │   │   ├── storage/rtu_metadata.db # SQLite database with 1,534 questions
│   │   │   └── indexes/               # Precomputed FAISS & BM25 binary indexes
│   │   └── scripts/                   # Database seeding and OCR ingestion utilities
│   │
│   ├── fine_tuning/
│   │   ├── generate_expanded_dataset.py # Synthesizes 748 ChatML pairs from SQLite
│   │   ├── rtu_train_dataset_expanded.jsonl # 748-example fine-tuning dataset
│   │   ├── RTU_Senior_Tutor_FineTuning_Colab.ipynb # Free Google Colab T4 training notebook
│   │   ├── train_qlora_colab.py       # Colab QLoRA training script
│   │   └── validate_dataset.py        # Dataset validation script
│   │
│   └── models/
│       ├── Modelfile                  # Ollama configuration for rtu-senior
│       ├── rtu_senior_tutor_q8.gguf   # 8-bit quantized GGUF model (1.57 GB)
│       └── rtu_senior_tutor_q4.gguf   # 4-bit quantized GGUF model (940 MB)
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
All past question papers and university syllabus structures referenced in this project are the intellectual property of Rajasthan Technical University (RTU), Kota.
