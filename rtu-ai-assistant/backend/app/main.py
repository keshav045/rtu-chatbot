"""
FastAPI Application Entry Point for RTU AI Academic Assistant
Provides clean lifespan initialization, structured logging, CORS, and request timing.
"""

import sys
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure backend package resolution
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.database.session import SessionLocal
from app.retrieval.pipeline import RetrievalPipeline
from app.services.chat_service import ChatService
from app.core.tracer import setup_langsmith_tracing
from app.retrieval.lc_retriever import build_retriever
from app.services.analytics_service import AnalyticsService
from app.services.importance_service import ImportanceService
from app.services.planner_service import PlannerService
from app.services.quiz_service import QuizService
from app.services.upload_service import UploadService
from app.api.endpoints import router as api_router, init_services

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("rtu.app")

# Lifespan manager: Loads models and indexes once on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing RTU AI Academic Assistant Backend...")
    setup_langsmith_tracing()
    db = SessionLocal()
    try:
        pipeline = RetrievalPipeline(index_dir=settings.INDEXES_DIR)
        pipeline.load_indexes(db)
        logger.info("Hybrid Retrieval Engine (FAISS + BM25 + CrossEncoder) ready.")

        logger.info("Initializing LangChain Hybrid Compression Retriever...")
        lc_retriever = build_retriever(db=db)
        logger.info("LangChain Retriever initialized.")

        analytics_service = AnalyticsService(dense_indexer=pipeline.dense_indexer)
        chat_service = ChatService(pipeline=pipeline, lc_retriever=lc_retriever)
        importance_service = ImportanceService()
        planner_service = PlannerService()
        quiz_service = QuizService()
        upload_service = UploadService()

        init_services(
            chat_svc=chat_service,
            analytics_svc=analytics_service,
            pipeline=pipeline,
            importance_svc=importance_service,
            planner_svc=planner_service,
            quiz_svc=quiz_service,
            upload_svc=upload_service
        )
        logger.info("Conversational RAG, Crash Planner, Quiz & Crowdsource Upload services initialized.")
    except Exception as e:
        logger.error(f"Startup initialization error: {e}", exc_info=True)
        raise e
    finally:
        db.close()

    logger.info("RTU AI Academic Assistant Backend is ready to serve requests.")
    yield
    logger.info("Shutting down RTU AI Academic Assistant Backend...")

# OpenAPI Metadata
tags_metadata = [
    {
        "name": "System & Health",
        "description": "System health checks, database status, and operational metrics.",
    },
    {
        "name": "Conversational RAG",
        "description": "Multi-turn RAG chat with contextual query rewriting and verified citations.",
    },
    {
        "name": "Search & Retrieval",
        "description": "Hybrid semantic (FAISS) and keyword (BM25) search with Cross-Encoder reranking.",
    },
    {
        "name": "Paper Archives & PDF Streaming",
        "description": "Direct access to RTU past examination papers and raw PDF streams.",
    },
    {
        "name": "Academic Intelligence",
        "description": "Repeated question detection, cross-year comparisons, and curriculum analytics.",
    },
]

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Conversational intelligence and hybrid RAG system for Rajasthan Technical University past question papers.",
    openapi_tags=tags_metadata,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Logging Middleware
@app.middleware("http")
async def add_process_time_and_log(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    
    # Do not clutter logs with health checks
    if not request.url.path.endswith("/health"):
        logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({duration_ms:.1f}ms)")
    return response

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred processing your academic request.",
            "detail": str(exc)
        }
    )

# Mount REST API
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/api-info", tags=["System & Health"])
def api_info():
    return {
        "app": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs_url": "/docs",
        "api_prefix": settings.API_PREFIX
    }

# Mount static pure HTML/CSS/JS frontend
FRONTEND_DIR = settings.ASSISTANT_ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
