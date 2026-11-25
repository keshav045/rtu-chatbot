# RTU AI Academic Assistant - Comprehensive System & Script Reference

This document provides complete technical documentation for all primary scripts, services, contracts, and change-impact rules within the RTU AI Academic Assistant repository.

---

## Mandatory Workflow Rule (Read Before Every Edit, Update After Every Change)

1. **Before making any code, config, index, DB, or UI change**:
   - Read both `gemini.md` (this file) and `history.txt` to review existing contracts, cross-module dependencies, database constraints, and recent changes/regressions.
2. **After making any change**:
   - Update `gemini.md` whenever a script's purpose, CLI/runtime behavior, inputs/configuration, outputs/side effects, or cross-module contract changes.
   - Prepend an entry to `history.txt` recording: **Date & Timestamp**, **Commit Hash** (or `UNCOMMITTED (Working Tree)` before committing), **Files Changed**, **Change Details**, and **Cascading Impacts** (across modules, databases, retrieval indexes, UI, API contracts, and tests).

---

## 1. FastAPI Application Server

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/main.py`
- **Core Purpose & Role**:
  The central backend application entrypoint. Initializes the FastAPI REST application, mounts CORS middleware, runs `init_db()` during startup lifespan to ensure all SQLAlchemy tables (including `student_memories` and `topic_mastery`) exist, preloads retrieval pipelines (FAISS index, BM25 index, BGE embedding model, Cross-Encoder reranker, LangChain hybrid retriever), injects `pipeline` into `UploadService`, mounts REST endpoints under `/api`, and serves a 200 HTML landing page at `GET /` when `frontend/` does not exist.
- **Execution Command & Runtime Environment**:
  - Command: `python -u -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (executed from `rtu-ai-assistant/backend/`)
  - Runtime: Python 3.11 with FastAPI, Uvicorn, SQLAlchemy, LangChain, FAISS, PyTorch, SentenceTransformers.
- **Key Inputs & Configuration Parameters**:
  - `PORT` (default `8000`), `HOST` (`127.0.0.1` / `0.0.0.0`).
  - Environment variables from `.env`: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `DEFAULT_LLM_PROVIDER`, `DATABASE_URL`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`.
- **Outputs & Side Effects**:
  - Serves HTTP REST endpoints on `/api/chat`, `/api/search`, `/api/health`, `/api/subjects`, `/api/topics/{id}/important`, `/api/plan`, `/api/quiz/*`, `/api/uploads`, `/api/papers*`, `/api/analytics/*`, `/api/auth/*`, `/api/feedback*`, `/api/observability/*`.
  - Side Effects: Calls `init_db()` on `rtu_metadata.db`, loads ~1.5 GB into memory for embedding/reranker models, initializes `lc_faiss` on disk if missing.
- **End-to-End Operational Use Case**:
  Core backend service queried by Streamlit UI (`streamlit_app.py`), `benchmark/rtu_benchmark.py`, and REST clients.

---

## 2. Streamlit Student Interface

- **Script Path & File Name**: `streamlit_app.py` (with theme config in `.streamlit/config.toml`; launched locally via `run_app.bat` or `streamlit run streamlit_app.py --server.port 8501`)
- **Core Purpose & Role**:
  Full-screen dark charcoal Gemini/ChatGPT-style student interface that opens directly into the chat canvas (zero startup auth block). Features right-aligned user prompt pills, clean unboxed assistant responses, bottom `st.chat_input`, collapsible source citations, compact white inline `:material/thumb_up:` / `:material/thumb_down:` / `Copy` (`render_assistant_footer`, `render_copy_icon_button`) icons, **New chat**, strictly **Stream**, **Year**, and **Subject** selectors, per-user/guest conversation history with load, single-chat delete (`✕`), and clear-all (`🗑`) controls, and **Log In** / **Sign Up** modal triggers (`@st.dialog`) pinned at the bottom-left of the sidebar (switching to user badge + **Log Out** once authenticated).
- **Execution Command & Runtime Environment**:
  - Command: `streamlit run streamlit_app.py --server.port 8501` (executed from repository root; `run_app.bat` sets `BACKEND_URL=http://127.0.0.1:8000`).
  - Runtime: Python 3.11 with Streamlit (`1.65+`), Requests.
- **Key Inputs & Configuration Parameters**:
  - `BACKEND_URL` env var (defaults to Cloud URL unless `run_app.bat` / env sets `http://127.0.0.1:8000`); `.streamlit/config.toml` sets dark theme colors (`#0b0b0c` canvas, `#171718` sidebar).
  - Student auth modal credentials (`roll_no`, `email`, `name`, `branch`, `year`, `password`), sidebar selectors (`Stream`, `Year`, dynamically filtered `Subject` via `GET /api/subjects?branch=...&year=...` with `FALLBACK_SUBJECTS`), chat input (`st.chat_input`), and `👍` / `👎` votes.
- **Outputs & Side Effects**:
  - Generates an isolated session `guest_user_id` before login and switches to authenticated `user_id` via `/api/auth/login` or `/api/auth/register` (resolving 4xx validation/auth errors and nested `error.message` payloads cleanly via `_extract_error_detail()`, while `api_request()` rejects unexpected `text/html` proxy pages cleanly).
  - Streams assistant responses live token-by-token via `POST /api/chat/stream` (`_iter_chat_sse`, updating an in-place markdown placeholder with `▌` cursor and falling back cleanly to `POST /api/chat`).
  - Scopes conversations and chat requests by `active_user_id()` (`GET /api/conversations?user_id=...`, `GET /api/conversations/{id}`, `DELETE /api/conversations/{id}`, `DELETE /api/conversations?user_id=...`) and sends top-level `branch`, `year`, `user_id`, and optional `filters.subject` to `POST /api/chat`.
  - Records inline correctness feedback via `POST /api/feedback`.
- **End-to-End Operational Use Case**:
  Students land straight into the dark workspace to ask RTU exam questions immediately by Stream, Year, and Subject, load or delete past chats anytime, and optionally sign up or log in from the bottom-left sidebar bar for persistent account history.

---

## 3. Unified Retrieval Pipeline & LC Retriever

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/retrieval/pipeline.py` (paired with `topic_verifier.py`, `lc_retriever.py`, `dense_indexer.py`, `bm25_indexer.py`, `reranker.py`, `hybrid_retriever.py`, `context_builder.py`)
- **Core Purpose & Role**:
  Executes hybrid retrieval combining dense vector search (`FAISS` + `BAAI/bge-small-en-v1.5` explicitly on `device="cpu"`, with 1.25x subject score boost when `subject` and `id_mapping_meta` are passed) and sparse keyword matching (`BM25Okapi`), filters by academic metadata (`subject`, `semester`, `year_from`, `year_to`, `section`, `marks`), fuses rankings via Reciprocal Rank Fusion (`k=60`), reranks candidates via `cross-encoder/ms-marco-MiniLM-L-6-v2` on `device="cpu"` (marking `rerank_fallback=True` without inflating logits if unavailable) with MMR diversification, and caches responses in an LRU cache (`maxsize=200`). Paired `TopicPaperVerifier` (`topic_verifier.py`) normalizes filler words, domain shorthands (`btree` -> `B-Tree`), and domain typos (`deadlovk` -> `deadlock`), embeds and caches all `1,182` RTU syllabus `Topic` rows (`topic_embeddings.npz`) to verify query-to-RTU relevance before generation (even for subjects with 0 extracted past-paper questions), and retrieves strongly related syllabus topics + CrossEncoder-ranked topic-linked questions.
- **Execution Command & Runtime Environment**:
  - Invoked via `RetrievalPipeline.retrieve(...)`, `RetrievalPipeline.verify_and_retrieve(...)`, or via `build_retriever(db)` in `lc_retriever.py`.
  - Runtime: PyTorch, SentenceTransformers, Rank-BM25, FAISS, LangChain.
- **Key Inputs & Configuration Parameters**:
  - `query`, `subject`, `semester`, `year_from`, `year_to`, `section`, `marks`, `top_k=5`, `rerank_candidates_k=12`.
  - LRU cache key: `(query_normalized, subject, semester, year_from, year_to, section, marks, top_k)`.
- **Outputs & Side Effects**:
  - `retrieve()` returns `results` (with CrossEncoder `rerank_score`), `citations`, `context_string`, and `retrieval_stats`.
  - `verify_and_retrieve()` returns `is_rtu_related`, `related_topics`, `questions`, `citations`, and `topic_eval` diagnostics.
- **End-to-End Operational Use Case**:
  Called directly by `/api/search` and `ChatService` (directly for filtered queries and as CrossEncoder rescorer/fallback alongside `lc_retriever`).

---

## 4. Chat Service & Guardrail Orchestrator

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/services/chat_service.py`
- **Core Purpose & Role**:
  Central RAG and exam coaching orchestrator. Handles session resolution with user upsert (`_ensure_user_exists`), multi-turn history & query rewriting (`QueryRewriter`), security & syllabus screening (`AcademicScreeningService`), intent dispatch (`STUDY_PLAN`, `TOPIC_IMPORTANCE`, `PAST_YEAR_QUESTIONS`, `QUIZ_REQUEST`, `SUBJECT_SELECT`, `OUT_OF_SYLLABUS`, `EXAM_QUESTION`), pre-generation RTU topic & past-paper verification (`pipeline.verify_and_retrieve()`, rejecting off-topic queries even on unindexed subjects like `OS`/`PHY`), exhaustive related-topic & topic-linked question retrieval, long-term student memory injection (`MemoryService`), mark-aware anti-boilerplate LLM synthesis using `DEFAULT_LLM_PROVIDER="auto"` fallback (`resolve_auto_providers`: `openai` -> `groq` -> `gemini` -> `ollama` `qwen2.5:1.5b` -> `offline`), anti-leakage post-sanitization, and LangSmith tracing.
- **Execution Command & Runtime Environment**:
  - Invoked via `ChatService.process_turn(...)`, `ChatService.stream_turn(...)` (for SSE token streaming), or `ChatService.chat(...)`.
  - Runtime: FastAPI backend dependency.
- **Key Inputs & Configuration Parameters**:
  - `db`, `user_message`, `session_id`, `provider_name`, `filters`, `top_k`, `language_pref`, `mode`, `user_id`, `branch`, `year`.
  - Subject-count check: flags `is_unindexed_subject=True` when `target_subject` has 0 papers OR 0 approved questions (e.g., `OS`), still running syllabus-topic verification via `TopicPaperVerifier` rather than skipping screening.
- **Outputs & Side Effects**:
  - Persists `User` (if unknown `user_id`), `SessionModel` (updating `updated_at` on each turn so active conversations sort first in `/api/conversations`), `Message` (`user` + `assistant`), `Citation`, and updates `StudentMemory`/`TopicMastery` in SQLite (skipping `MemoryService.record_turn_interaction()` when a turn is rejected for `SECURITY_VIOLATION`, `OUT_OF_SYLLABUS`, or `TopicPaperVerifier` `OUT_OF_SYLLABUS_TOPIC`); returns `related_topics` and `topic_verification` alongside `citations`; emits LangSmith traces.
- **End-to-End Operational Use Case**:
  Processes all conversational turns from Streamlit, `/api/chat`, benchmarks, and evaluation scripts while blocking prompt injection, off-topic queries, and cross-subject citation leaks.

---

## 5. Correctness & Rigor Benchmark Suite

- **Script Path & File Name**: `benchmark/rtu_benchmark.py` (with wrapper `benchmark/rtu.py` and rubric `benchmark/rubric.json`)
- **Core Purpose & Role**:
  Evaluates live chat responses across 12 standardized questions (`Q01`-`Q09` academic/planning/quiz + `Q10`-`Q12` out-of-syllabus refusals) for rubric concept groups, forbidden phrases, template boilerplate (`TEMPLATE_PHRASES`), unverified citations, length floor, duplicate answers, and refusal subject mismatches.
- **Execution Command & Runtime Environment**:
  - Live Run: `python benchmark/rtu_benchmark.py --rubric benchmark/rubric.json`
  - Dry-Run Scorer: `python benchmark/rtu_benchmark.py --dry-run benchmark/results/latest.json`
- **Key Inputs & Configuration Parameters**:
  - `--base-url` (`http://127.0.0.1:8000`), `--endpoint` (`/api/chat`), `--rubric` (`benchmark/rubric.json`), `--dry-run`.
  - Sends requests using `"session_id": "benchmark"` across all 12 prompts.
- **Outputs & Side Effects**:
  - Prints per-question diagnostics + summary metrics (`pass_rate`, `content_pass_rate`, `template_phrase_hits`, `unverified_citations`, `errors`) and writes timestamped JSON under `benchmark/results/`.
- **End-to-End Operational Use Case**:
  Regression and LLM quality gate comparing base models vs fine-tuned weights (`rtu-senior`). Note: baseline `rtu-senior` (1.5B QLoRA) still exhibits model-level meta-rubric boilerplate and factual hallucinations on `Q01`-`Q06`/`Q09` (`5/12` pass after backend/routing fixes).

---

## 6. Document Preparer & Index Builder

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/retrieval/build_indexes.py` & `rtu-ai-assistant/backend/app/retrieval/document_preparer.py`
- **Core Purpose & Role**:
  Extracts approved questions from `rtu_metadata.db`, filters out malformed OCR fragments/headers via `is_meaningful_question()` (removing empty headers, page footers, and non-text math/number noise while preserving short valid 2-mark Part A questions such as `"Define SRS."`), builds BGE-Small FAISS embeddings (`rtu_faiss_bge.index`), builds `BM25Okapi` index (`bm25_index.pkl`), and saves `faiss_id_mapping.json`.
- **Execution Command & Runtime Environment**:
  - Command: `python -m app.retrieval.build_indexes` (executed from `rtu-ai-assistant/backend/`).
- **Key Inputs & Configuration Parameters**:
  - Source DB: `backend/data/storage/rtu_metadata.db` (724 papers, 1,534 approved questions).
  - Target Directory: `backend/data/indexes/`.
- **Outputs & Side Effects**:
  - Generates `rtu_faiss_bge.index`, `faiss_id_mapping.json`, `bm25_index.pkl` for **1,493** clean searchable questions. Delete `backend/data/indexes/lc_faiss/` before restarting FastAPI if `lc_faiss` should also rebuild from the updated document set.
- **End-to-End Operational Use Case**:
  Run offline after DB ingestion/cleaning to keep dense, sparse, and LangChain vector stores synchronized with SQLite.

---

## 7. Fine-Tuning Pipeline (Unsloth QLoRA)

- **Script Path & File Name**: `rtu-ai-assistant/fine_tuning/train_qlora_colab.py` (and `models/Modelfile`)
- **Core Purpose & Role**:
  Google Colab / GPU script that fine-tunes Qwen2.5-3B-Instruct using Unsloth 4-bit QLoRA on a curated 748-example RTU Senior Tutor dataset (`rtu-senior`).
- **Execution Command & Runtime Environment**:
  - Run in Google Colab (T4 / A100 GPU) or via `python fine_tuning/train_qlora_colab.py`.
- **Key Inputs & Configuration Parameters**:
  - Dataset: `rtu-ai-assistant/fine_tuning/seed_dataset_200.jsonl` / `rtu_train_dataset_expanded.jsonl`.
  - LoRA rank $r=16$, $\alpha=32$, Learning Rate $2 \times 10^{-4}$, Epochs 3.
- **Outputs & Side Effects**:
  - Trained LoRA adapters and GGUF weights deployed to Ollama (`rtu-senior:latest`).
- **End-to-End Operational Use Case**:
  Supplies domain-tuned local Ollama model weights for `ChatService` when `DEFAULT_LLM_PROVIDER="ollama"`.

---

## 8. Observability & Tracing Service

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/core/tracer.py`
- **Core Purpose & Role**:
  Provides LangSmith tracing (`start_trace`, `end_trace`), feedback sync (`send_feedback_to_langsmith`), and guardrail evaluation logging (`send_screening_eval_to_langsmith`). Includes a `_FEEDBACK_WRITE_DISABLED` circuit breaker that disables further HTTP `/feedback` posts after the first `403 Forbidden` response so read-only keys do not repeatedly spam logs or add turn latency.
- **Execution Command & Runtime Environment**:
  - Imported by `main.py`, `endpoints.py`, and `chat_service.py`.
- **Key Inputs & Configuration Parameters**:
  - `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`, `LANGCHAIN_ENDPOINT`.
- **Outputs & Side Effects**:
  - Asynchronous trace creation/updates to LangSmith API (`https://api.smith.langchain.com`).
- **End-to-End Operational Use Case**:
  Traces chat latency and screening scores per turn cleanly with zero blocking failures on restricted API keys.

---

## 9. Long-Term Student Memory Service

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/services/memory_service.py`
- **Core Purpose & Role**:
  Provides persistent cross-session student memory stored in `users`, `student_memories`, and `topic_mastery`: tracks academic goals/exam timelines, autonomously flags weak areas from struggle phrasing (`"struggling with..."`, `"confused about..."`), boosts confidence on mastery signals or quiz scores, builds the `[STUDENT LONG-TERM ACADEMIC MEMORY]` prompt injection block, and supports explicit memory CRUD/reset.
- **Execution Command & Runtime Environment**:
  - Invoked via `MemoryService.get_or_create_user`, `get_student_memory_context`, `record_turn_interaction`, `update_topic_mastery`, `clear_student_memory`.
- **Key Inputs & Configuration Parameters**:
  - `db`, `user_id` (defaults to `"student_default"`), `target_subject` (matched case-insensitively with exact `func.upper(TopicMastery.subject_code) == clean_sub` so short codes like `DS` do not match `DSA`/`DBMS`).
- **Outputs & Side Effects**:
  - Reads/writes `User`, `StudentMemory`, and `TopicMastery` rows in `rtu_metadata.db`; injects memory context into `ChatService` system prompts.
- **End-to-End Operational Use Case**:
  Enables personalized tutoring across sessions where the Senior Tutor remembers weak topics, mastered concepts, and upcoming exam goals.

---

## 10. FastAPI REST Endpoints & Answer Correctness Tracker

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/api/endpoints.py`
- **Core Purpose & Role**:
  Exposes all HTTP API contracts: sessions, synchronous & SSE streaming chat (`/api/chat`, `/api/chat/stream`), hybrid search, important topics, crash study planner, quiz workflow, crowdsourced PDF uploads/review queue, paper catalog & PDF streaming, student auth (`/api/auth/register` resolves branch codes like `"CSE"` -> `Branch.id` `"BRANCH_CSE"`), student long-term memory (`/api/memory*`), analytics, observability, and the **Feedback & Answer Correctness Tracker** (`/api/feedback`).
- **Execution Command & Runtime Environment**:
  - Mounted under `/api` by `app/main.py`.
- **Key Inputs & Configuration Parameters**:
  - Pydantic schemas: `ChatRequest` (top-level `branch`, `year`, `filters`), `ChatFilterRequest`, `FeedbackRequest`, `PlanRequest` (`hours_left`), `RegisterRequest`, `QuizNextRequest`, `QuizAnswerRequest` (optional `user_id`), `MemoryCreateRequest`, `TopicMasteryUpdateRequest`.
  - Query filters on `GET /api/feedback`: `verdict` (`correct`/`incorrect`), `subject`, `limit`; `/api/papers` uses `.distinct()` to prevent duplicate rows across `SubjectOffering` branches.
- **Outputs & Side Effects**:
  - Auth endpoints (`/api/auth/register`, `/api/auth/login`, `/api/auth/me`) return normalized user profiles via `_format_user_profile()` so `branch` always exposes the branch code (e.g. `"CSE"`) rather than `"BRANCH_CSE"`.
  - Per-user history & deletion routes (`GET /api/conversations?user_id=...`, `GET /api/conversations/{id}`, `DELETE /api/conversations/{id}`, `DELETE /api/conversations?user_id=...`) filter and delete user-owned conversations.
  - Student Long-Term Memory routes (`GET /api/memory`, `POST /api/memory`, `POST /api/memory/mastery`, `DELETE /api/memory/{id}`, `DELETE /api/memory`) expose persistent `StudentMemory` facts/goals and `TopicMastery` records.
  - `POST /api/feedback` upserts by `message_id` and returns `is_correct` (`bool`) & `verdict` (`"Correct"` / `"Incorrect"`).
  - `GET /api/feedback` resolves associated `Message` + preceding user question + `SessionModel` subject via `_resolve_feedback_qa()` and returns `{summary: {total_rated, correct_count, incorrect_count, accuracy_pct}, records: [...]}`.
  - `DELETE /api/feedback/{feedback_id}` deletes a tracked feedback record.
- **End-to-End Operational Use Case**:
  Powers student authentication, per-user conversation history, chat RAG, inline `Copy / 👍 / 👎` controls, and backend API features.

---

## 11. Intent Router, Importance Rankings & Quiz Services

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/services/intent_service.py`, `importance_service.py`, `quiz_service.py`, `planner_service.py`
- **Core Purpose & Role**:
  - `IntentRouter`: classifies `SECURITY_VIOLATION`, `GREETING`, `CLOSING`, `HELP_CAPABILITIES`, `STUDY_PLAN` (supporting plural `hours?`/`hrs?`), `TOPIC_IMPORTANCE`, `PAST_YEAR_QUESTIONS`, `QUIZ_REQUEST`, `SUBJECT_SELECT`, `OUT_OF_SYLLABUS`, and `EXAM_QUESTION`.
  - `ImportanceService`: ranks topics by SQL mark/year aggregation (`Tier A`/`B`) and falls back cleanly via `_build_syllabus_fallback()` when a cataloged subject has >0 papers but 0 extracted questions (e.g., `OS`, `CN`, `AI`, `DAA`).
  - `QuizService`: serves practice questions and evaluates answers on the 4-pillar RTU rubric (`rubric_breakdown`), verifying `SessionModel` existence before saving `QuizAttempt` and updating persistent `TopicMastery` confidence/`is_weak_area` via `MemoryService` (`_resolve_quiz_context` + `_record_quiz_mastery`).
- **Execution Command & Runtime Environment**:
  - Programmatic service classes called by `endpoints.py` and `chat_service.py`.
- **Key Inputs & Configuration Parameters**:
  - `subject_id`, `hours_left`/`hours_available`, `question_id`, `student_answer`, `target_marks`.
- **Outputs & Side Effects**:
  - Deterministic routing, topic weightages, knapsack revision schedules, and rubric scores (`QuizAttempt` inserts require valid `SessionModel.id`).
- **End-to-End Operational Use Case**:
  Supports the Crash Course Planner, Important Topics dashboard, Practice Quiz grader, and 3-question quiz chat mode.

---

## 12. Crowdsourced Upload & Incremental Index Sync Service

- **Script Path & File Name**: `rtu-ai-assistant/backend/app/services/upload_service.py`
- **Core Purpose & Role**:
  Validates student PDF uploads with SHA-256 deduplication, extracts question text (`_extract_questions_from_pdf`), persists `Paper` and `Question` rows (`_persist_paper_and_questions`) on both auto-approval and admin review-queue approval, and incrementally updates `RetrievalPipeline` (cache clear, `doc_store`, `BM25Indexer`, FAISS vector append, and `lc_retriever._CACHED_RETRIEVER` reset) via `_sync_retrieval_indexes()`.
- **Execution Command & Runtime Environment**:
  - Instantiated in `main.py` as `UploadService(pipeline=pipeline)` and exposed via `/api/uploads` & `/api/admin/review/*`.
- **Key Inputs & Configuration Parameters**:
  - PDF bytes, `subject_code`, `year`, `action` (`"approve"` / `"reject"`).
- **Outputs & Side Effects**:
  - Writes PDF files to `backend/data/storage/student_uploads/`, inserts `Upload`/`Paper`/`Question` rows in SQLite, and refreshes live FAISS/BM25 indexes.
- **End-to-End Operational Use Case**:
  Allows students and reviewers to contribute new RTU past papers and make extracted questions searchable without server restarts.

---

## 13. Automated Test Suite

- **Script Path & File Name**: `rtu-ai-assistant/backend/tests/` (`test_phase1_ingestion.py`, `test_phase2_retrieval.py`, `test_phase3_api.py`, `test_phase5_security_and_regression.py`, `test_screening_and_langsmith.py`, `test_v2_features.py`, `test_student_long_term_memory_e2e.py`, `test_feedback_tracking.py`, `test_topic_verification_retrieval.py`, `test_streaming_chat.py`, and CLI smoke script `verify_all_fixes.py`)
- **Core Purpose & Role**:
  Validates database integrity, ingestion manifest, FAISS/BM25/CrossEncoder retrieval, RTU topic-paper query verification, API contracts, security guardrails, LangSmith screening, student long-term memory, and Answer Correctness feedback tracking.
- **Execution Command & Runtime Environment**:
  - Full Pytest Suite: `python -m pytest -o pythonpath=backend backend/tests -v` (executed from `rtu-ai-assistant/`, `43 passed`)
  - Application Eval: `python rtu-ai-assistant/evaluation/eval_application.py` (`4/4 PASS`)
  - Live Smoke Script: `python rtu-ai-assistant/backend/tests/verify_all_fixes.py` (with server running on `:8000`, `6/6 PASS`)
- **Key Inputs & Configuration Parameters**:
  - Uses local `rtu_metadata.db`, built indexes under `backend/data/indexes/`, and isolated `:memory:` SQLite for `test_student_long_term_memory_e2e.py`.
- **Outputs & Side Effects**:
  - Cleans up test records (`test_feedback_tracking.py`, `test_phase5`); Note `test_phase1_ingestion.py` asserts exact manifest field `total_papers_cataloged == 123`.
- **End-to-End Operational Use Case**:
  Verification gate before any commit or deployment.

---

## 14. Hugging Face Space Gradio + FastAPI Combined Entrypoint

- **Script Path & File Name**: `app.py` (with root `requirements.txt` and `README.md`)
- **Core Purpose & Role**:
  Serves both the Gradio web interface (`/` with `"Auto (Best Available)"` provider default) and the full FastAPI REST router (`/api/*`) on Hugging Face Spaces (`sdk: gradio`, `sdk_version: 6.32.0`). Satisfies HF ZeroGPU startup (`@spaces.GPU` probe) while forcing explicit `device="cpu"` on `DenseIndexer`, `CrossEncoderReranker`, and `lc_retriever` (`HuggingFaceEmbeddings` / `HuggingFaceCrossEncoder`) so ZeroGPU CUDA hooks never corrupt retrieval inside FastAPI requests. Runs `setup_langsmith_tracing()`, `init_db()`, `RetrievalPipeline`, `lc_retriever`, and `UploadService(pipeline=pipeline)` cleanly at startup, opens per-request `SessionLocal()` DB sessions in Gradio callbacks, and mounts `api_router` onto `gradio.routes.App` BEFORE `demo.launch(..., ssr_mode=False, strict_cors=False, _app=gradio_app)` so Gradio 6 SSR/SPA routing never blocks `POST /api/chat` with `405 Method Not Allowed`.
- **Execution Command & Runtime Environment**:
  - Command: `python app.py` (or `python app.py --test` for initialization smoke check).
  - Port: `PORT` env var (default `7860`).
- **Key Inputs & Configuration Parameters**:
  - Root `requirements.txt` (includes `langchain*`, `langsmith`, `openai`, `python-dotenv`, and `gradio>=6.0.0`); HF Space Secrets (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `LANGCHAIN_API_KEY`).
- **Outputs & Side Effects**:
  - Serves `GET /` (Gradio UI) and `/api/*` REST endpoints on port `7860`.
- **End-to-End Operational Use Case**:
  Powers `https://keshavagarwal779-rtu-exam-assistant.hf.space`, enabling external UIs like `streamlit_app.py` on Streamlit Cloud to call `/api/chat`, `/api/auth/*`, `/api/memory*`, and `/api/conversations*`.

---

## Cross-Module Impact & Pitfall Notes

1. **`ingestion_manifest.json` vs `test_phase1_ingestion.py`**: `test_phase1_ingestion.py` strictly asserts `manifest["total_papers_cataloged"] == 123` and `manifest["total_questions_ingested"] > 300`. Keep `total_papers_cataloged` at `123` and record updated totals in `total_papers_in_database` (`724`) and `total_questions_ingested` (`1493`).
2. **SQLite Foreign Keys (`PRAGMA foreign_keys=ON`)**: Always enabled on every connection (`session.py`).
   - Creating `SessionModel` with `user_id` requires an existing `User` row (`ChatService._ensure_user_exists`).
   - Creating `User` with `branch_id` requires the full `branches.id` key (e.g. `"BRANCH_CSE"`, NOT `"CSE"`).
   - Creating `QuizAttempt` requires an existing `SessionModel.id`.
   - Deleting a session must clear `citations` or cascade cleanly (`Message.citations` has `delete-orphan`, `QuizAttempt` has FK to `sessions.id`).
3. **`Base.metadata.create_all()` limitation**: `init_db()` creates missing tables (`student_memories`, `topic_mastery`) on startup, but will NOT add columns to already-existing SQLite tables (`feedbacks`, `messages`, etc.).
4. **CrossEncoder Score Thresholds (`config.py`)**: With `ms-marco-MiniLM-L-6-v2`, out-of-scope queries score `< -10.5` while valid partially-covered RTU queries (e.g., `"TCP three-way handshake"`) score around `-8.0`. Refusal cutoff is `-9.5` (`OUT_OF_SCOPE_THRESHOLD`) and citation confidence cutoff is `-6.0` (`CITATION_RELEVANCE_THRESHOLD`). Any change to `lc_retriever` must still re-score docs with `pipeline.reranker.rerank()` because LangChain's `CrossEncoderReranker` does not populate `relevance_score` in `Document.metadata`.
5. **Subject Isolation vs Multi-Turn Follow-ups (`chat_service.py`)**: Standalone `EXAM_QUESTION` queries without UI `filters.subject` must NOT inherit prior conversation turn subjects (only follow-ups, clarifications, or explicit commands inherit history subject), or multi-subject sessions/benchmarks get trapped on prior topics.
6. **UI / API Key Contracts**:
   - `/api/topics/{id}/important` returns `years_appeared` (not `paper_appearances`).
   - `/api/quiz/next` returns `target_marks`, `part`, `source` (not `max_marks`, `unit_no`).
   - `/api/quiz/answer` returns `rubric_breakdown` (not `breakdown`).
   - `/api/plan` expects `hours_left` (not `hours_available`).
   - `/api/chat` expects top-level `branch` and `year`.
