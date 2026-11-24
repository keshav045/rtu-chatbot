"""
FastAPI REST API Endpoints for RTU AI Academic Assistant
Comprehensive implementation of Section 10 of Implementation Plan Version 2:
- Smart Onboarding & Session Management
- Data-Backed Important Topics (Tier A/B/C)
- Crash-Course Knapsack Planner (10% Buffer + Formula Revision)
- Practice Quiz & Rubric Evaluation (0-10 Score + Senior Feedback)
- Crowdsourced PDF Upload Pipeline (SHA-256 Deduplication & Review Queue)
- Senior Tutor Conversational RAG (Hinglish/English & 10-Mark Answer Coach)
- Original Paper Archives & PDF Streaming
"""

import logging
import uuid
import hashlib
import hmac
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.session import get_db
from app.database.models import (
    Paper, Question, Subject, Unit, Topic, SessionModel, Message,
    Citation, Upload, Feedback, QuizAttempt, Branch, SubjectOffering, User
)
from app.core.config import settings
from app.core.tracer import send_feedback_to_langsmith, is_tracing_active, get_current_project_name
from app.core.langsmith_datasets import sync_benchmark_to_langsmith

AUTH_SALT = "rtu_exam_student_salt_v2"

def hash_password(password: str) -> str:
    return hashlib.sha256(f"{AUTH_SALT}_{password}".encode("utf-8")).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hmac.compare_digest(hash_password(plain_password), hashed_password)

logger = logging.getLogger("rtu.api")

router = APIRouter()

# Global service references
_chat_service = None
_analytics_service = None
_retrieval_pipeline = None
_importance_service = None
_planner_service = None
_quiz_service = None
_upload_service = None

def init_services(
    chat_svc,
    analytics_svc,
    pipeline,
    importance_svc=None,
    planner_svc=None,
    quiz_svc=None,
    upload_svc=None
):
    global _chat_service, _analytics_service, _retrieval_pipeline
    global _importance_service, _planner_service, _quiz_service, _upload_service
    _chat_service = chat_svc
    _analytics_service = analytics_svc
    _retrieval_pipeline = pipeline
    _importance_service = importance_svc
    _planner_service = planner_svc
    _quiz_service = quiz_svc
    _upload_service = upload_svc

# --- Pydantic Data Contracts ---

class SessionCreateRequest(BaseModel):
    subject_id: str = Field(..., description="Subject code or short name e.g. 'DSA', 'OOP', 'AEM'")
    branch: str = Field("CSE", description="Student branch e.g. 'CSE', 'IT', 'ECE', 'EE', 'ME', 'CE'")
    year: int = Field(2, ge=1, le=4, description="Year of study: 1, 2, 3, or 4")
    hours_left: float = Field(5.0, ge=0.5, le=100.0, description="Hours left before exam")
    mode: str = Field("explain_topic", description="Active mode: 'crash_course', 'explain_topic', 'practice_quiz', 'important_questions'")
    user_id: Optional[str] = None

class RegisterRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Full student name")
    roll_no: str = Field(..., min_length=2, max_length=50, description="RTU Roll Number or Student ID")
    email: Optional[str] = Field(None, max_length=100, description="Email address")
    password: str = Field(..., min_length=4, max_length=100, description="Password")
    branch: Optional[str] = Field("CSE", max_length=20, description="Branch of study")
    year: Optional[int] = Field(2, ge=1, le=4, description="Current year of study")
    college: Optional[str] = Field("Rajasthan Technical University (RTU)", max_length=150)

class LoginRequest(BaseModel):
    roll_no_or_email: str = Field(..., min_length=2, max_length=100, description="Roll Number or Email")
    password: str = Field(..., min_length=3, max_length=100, description="Password")

class ChatFilterRequest(BaseModel):
    subject: Optional[str] = None
    semester: Optional[int] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    section: Optional[str] = None
    marks: Optional[int] = None
    branch: Optional[str] = None
    year: Optional[int] = None

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Student's query or prompt")
    session_id: Optional[str] = Field(None, description="Active session ID")
    conversation_id: Optional[str] = Field(None, description="Backward compatibility alias")
    provider: Optional[str] = Field(None, description="LLM provider: 'offline', 'ollama', 'groq', 'gemini'")
    language: Optional[str] = Field(None, description="'english', 'hinglish', or None for auto")
    mode: Optional[str] = Field(None, description="'standard', 'answer_coach', 'crash', 'quiz'")
    branch: Optional[str] = Field(None, description="Student branch e.g. 'CSE', 'IT'")
    year: Optional[int] = Field(None, ge=1, le=4, description="Year of study 1-4")
    filters: Optional[ChatFilterRequest] = None
    top_k: int = Field(5, ge=1, le=20)
    user_id: Optional[str] = Field(None, description="Logged in student user ID")

class PlanRequest(BaseModel):
    subject_id: str = Field(..., description="Subject short name or code e.g. 'DSA'")
    hours_left: float = Field(5.0, ge=0.5, le=100.0, description="Hours available for study")
    target_score: Optional[int] = Field(70, ge=30, le=100, description="Target percentage or marks")
    student_level: Optional[str] = Field("year2", description="Student level")

class QuizNextRequest(BaseModel):
    session_id: Optional[str] = None
    subject_id: str = Field(..., description="Subject code e.g. 'DSA'")
    topic_id: Optional[str] = None

class QuizAnswerRequest(BaseModel):
    session_id: Optional[str] = None
    question_id: str = Field(..., description="ID of the question being answered")
    student_answer: str = Field(..., min_length=2, description="Student's written answer")

class FeedbackRequest(BaseModel):
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    trace_id: Optional[str] = Field(None, description="LangSmith trace/run ID for direct feedback synchronization")
    rating: Any = Field(..., description="'up'/1 for thumbs up, 'down'/-1 for thumbs down")
    comment: Optional[str] = None

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    subject: Optional[str] = None
    semester: Optional[int] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    section: Optional[str] = None
    marks: Optional[int] = None
    top_k: int = Field(5, ge=1, le=30)

# --- System & Health Endpoints ---

@router.get("/health", tags=["System & Health"], summary="Service Health & Index Status")
def health_check(db: Session = Depends(get_db)):
    """Verifies database connectivity, paper/question counts, and index readiness."""
    paper_count = db.query(Paper).count()
    question_count = db.query(Question).count()
    subject_count = db.query(Subject).count()
    topic_count = db.query(Topic).count()
    return {
        "status": "healthy",
        "app_name": settings.PROJECT_NAME,
        "version": "2.0.0",
        "database": {
            "papers": paper_count,
            "questions": question_count,
            "subjects": subject_count,
            "topics": topic_count,
            "connected": True
        },
        "retrieval_loaded": _retrieval_pipeline._is_loaded if _retrieval_pipeline else False
    }

# --- Section 10: Smart Onboarding & Session Management ---

@router.post("/session", tags=["Session & Onboarding"], summary="Initialize Student Study Session")
def create_session(request: SessionCreateRequest, db: Session = Depends(get_db)):
    """
    Initializes a student study session with year, branch, subject, hours left, and mode.
    Returns session ID, subject data tier, and past paper coverage statistics.
    """
    subj = db.query(Subject).filter(
        (Subject.id == request.subject_id) |
        (func.lower(Subject.short_name) == request.subject_id.lower()) |
        (Subject.code == request.subject_id)
    ).first()

    if not subj:
        raise HTTPException(status_code=404, detail=f"Subject '{request.subject_id}' not found.")

    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")

    new_sess = _chat_service.create_session(
        db=db,
        subject_id=subj.id,
        branch=request.branch,
        year=request.year,
        hours_left=request.hours_left,
        mode=request.mode,
        user_id=request.user_id
    )

    paper_count = len(subj.papers)
    tier = subj.data_tier or ("Tier A" if paper_count >= 5 else ("Tier B" if paper_count >= 1 else "Tier C"))

    return {
        "session_id": new_sess.id,
        "subject_id": subj.id,
        "subject_name": subj.name,
        "subject_short": subj.short_name,
        "subject_code": subj.code,
        "branch": new_sess.branch,
        "year": new_sess.year,
        "hours_left": new_sess.hours_left,
        "mode": new_sess.mode,
        "tier": tier,
        "paper_count": paper_count,
        "disclaimer": (
            "Verified against 5+ years of RTU past papers." if tier == "Tier A"
            else ("Limited past papers available (1-4 years)." if tier == "Tier B"
            else "Syllabus-only mode. No past papers yet. Rankings based on syllabus weightage.")
        ),
        "created_at": new_sess.created_at.isoformat()
    }

@router.get("/session/{session_id}", tags=["Session & Onboarding"], summary="Get Session Detail & History")
def get_session(session_id: str, db: Session = Depends(get_db)):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    sess = _chat_service.get_session(db=db, session_id=session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return sess

@router.get("/sessions", tags=["Session & Onboarding"], summary="List Recent Study Sessions")
def list_sessions(
    limit: int = Query(50, ge=1, le=100),
    user_id: Optional[str] = Query(None, description="Optional user ID filter"),
    db: Session = Depends(get_db)
):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    return _chat_service.list_sessions(db=db, limit=limit, user_id=user_id)

# Backward-compatibility aliases for /conversations
@router.get("/conversations", tags=["Conversational RAG"], summary="List Conversations (Alias)")
def list_conversations_alias(
    limit: int = Query(50, ge=1, le=100),
    user_id: Optional[str] = Query(None, description="Optional user ID filter"),
    db: Session = Depends(get_db)
):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    return _chat_service.list_sessions(db=db, limit=limit, user_id=user_id)

# --- Student Authentication Endpoints ---

def _format_user_profile(user: User, message: Optional[str] = None) -> Dict[str, Any]:
    """Formats student auth/profile response resolving Branch.id ('BRANCH_CSE') -> code ('CSE')."""
    branch_code = user.branch.code if user.branch else (user.branch_id or "CSE").replace("BRANCH_", "")
    payload: Dict[str, Any] = {
        "status": "success",
        "user_id": user.id,
        "name": user.name,
        "roll_no": user.roll_no,
        "email": user.email,
        "branch": branch_code,
        "year": user.year,
        "college": user.college,
    }
    if message:
        payload["message"] = message
    return payload


@router.post("/auth/register", tags=["Student Authentication"], summary="Register New Student")
def register_student(request: RegisterRequest, db: Session = Depends(get_db)):
    """Registers a student account with RTU roll number, name, branch, and secure password."""
    # Check if user already exists with roll_no or email
    existing_roll = db.query(User).filter(func.lower(User.roll_no) == request.roll_no.strip().lower()).first()
    if existing_roll:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Student with roll number '{request.roll_no.strip()}' is already registered."
        )

    if request.email and request.email.strip():
        existing_email = db.query(User).filter(func.lower(User.email) == request.email.strip().lower()).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Student with email '{request.email.strip()}' is already registered."
            )

    # Resolve branch code ('CSE') to real Branch.id foreign key ('BRANCH_CSE')
    branch_obj = None
    if request.branch:
        branch_clean = request.branch.strip()
        branch_obj = db.query(Branch).filter(
            (Branch.id == branch_clean) | (func.lower(Branch.code) == branch_clean.lower())
        ).first()

    new_user = User(
        id=f"user_{uuid.uuid4().hex[:12]}",
        name=request.name.strip(),
        roll_no=request.roll_no.strip(),
        email=request.email.strip() if request.email else None,
        password_hash=hash_password(request.password),
        branch_id=branch_obj.id if branch_obj else None,
        year=request.year,
        college=request.college.strip() if request.college else "Rajasthan Technical University (RTU)"
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return _format_user_profile(new_user, message="Student registered successfully.")

@router.post("/auth/login", tags=["Student Authentication"], summary="Student Login")
def login_student(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticates student via roll number or email with password verification."""
    identifier = request.roll_no_or_email.strip().lower()
    user = db.query(User).filter(
        (func.lower(User.roll_no) == identifier) | (func.lower(User.email) == identifier)
    ).first()

    if not user or not user.password_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid roll number/email or password."
        )

    if not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid roll number/email or password."
        )

    return _format_user_profile(user, message="Login successful.")

@router.get("/auth/me", tags=["Student Authentication"], summary="Current Student Profile")
def get_current_user_profile(user_id: str = Query(..., description="Student User ID"), db: Session = Depends(get_db)):
    """Retrieves authenticated student profile by user ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student profile not found.")

    return _format_user_profile(user)

@router.get("/conversations/{conversation_id}", tags=["Conversational RAG"], summary="Get Conversation Detail (Alias)")
def get_conversation_alias(conversation_id: str, db: Session = Depends(get_db)):
    return get_session(session_id=conversation_id, db=db)

@router.delete("/conversations/{conversation_id}", tags=["Conversational RAG"], summary="Delete Conversation")
@router.delete("/session/{conversation_id}", tags=["Session & Onboarding"], summary="Delete Session")
def delete_conversation_alias(conversation_id: str, db: Session = Depends(get_db)):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    success = _chat_service.delete_session(db=db, session_id=conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": conversation_id}

@router.delete("/conversations", tags=["Conversational RAG"], summary="Clear All Conversations")
@router.delete("/sessions", tags=["Session & Onboarding"], summary="Clear All Sessions")
def delete_all_conversations(user_id: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    query = db.query(SessionModel)
    if user_id:
        query = query.filter(SessionModel.user_id == user_id)
    sessions = query.all()
    count = len(sessions)
    for s in sessions:
        db.delete(s)
    db.commit()
    return {"status": "all_deleted", "deleted_count": count}

@router.post("/conversations/{conversation_id}/clear", tags=["Conversational RAG"], summary="Clear Conversation")
def clear_conversation_alias(conversation_id: str, db: Session = Depends(get_db)):
    if not _chat_service:
        raise HTTPException(status_code=503, detail="Chat service not initialized")
    success = _chat_service.clear_session(db=db, session_id=conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "cleared", "session_id": conversation_id}

# --- Section 10: Important Topics Endpoint ---

@router.get("/topics/{subject_id}/important", tags=["Academic Intelligence"], summary="Get Ranked Important Topics")
def get_important_topics(
    subject_id: str,
    marks: Optional[int] = Query(None, description="Optional target marks (e.g. 10 for Part C, 2 for Part A)"),
    db: Session = Depends(get_db)
):
    """
    Returns data-driven topic ranking based on SQL aggregation over past RTU papers:
    - Years appeared
    - Long answer count (Marks >= 8)
    - Short answer count (Marks <= 2)
    - Total historical marks
    - Tier A / Tier B / Tier C transparent labeling
    """
    if not _importance_service:
        raise HTTPException(status_code=503, detail="Importance service not initialized")
    
    result = _importance_service.get_ranked_topics(db=db, subject_id=subject_id, target_marks=marks)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# --- Section 10: Crash-Course Planner Endpoint ---

@router.post("/plan", tags=["Academic Intelligence"], summary="Generate Crash-Course Study Plan")
def generate_study_plan(request: PlanRequest, db: Session = Depends(get_db)):
    """
    Generates a time-boxed crash course study schedule:
    - Knapsack optimization: picks topics maximizing marks_yield / study_minutes
    - 10% emergency buffer time
    - Dedicated final rapid revision block (formulas, definitions, diagrams)
    """
    if not _planner_service:
        raise HTTPException(status_code=503, detail="Planner service not initialized")
    
    result = _planner_service.generate_plan(
        db=db,
        subject_id=request.subject_id,
        hours_available=request.hours_left,
        student_level=request.student_level or "year2"
    )
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# --- Section 10: Practice Quiz & Answer Evaluation Endpoints ---

@router.post("/quiz/next", tags=["Quiz & Practice"], summary="Fetch Next Practice Question")
def get_next_quiz_question(request: QuizNextRequest, db: Session = Depends(get_db)):
    """Fetches a practice question for the subject/topic, prioritizing real past RTU questions."""
    if not _quiz_service:
        raise HTTPException(status_code=503, detail="Quiz service not initialized")
    
    return _quiz_service.get_next_question(
        db=db,
        subject_id=request.subject_id,
        topic_id=request.topic_id
    )

@router.post("/quiz/answer", tags=["Quiz & Practice"], summary="Submit Answer for 0-10 Rubric Evaluation")
@router.post("/quiz/evaluate", include_in_schema=False)
def evaluate_quiz_answer(request: QuizAnswerRequest, db: Session = Depends(get_db)):
    """
    Evaluates student answer on a 0-10 rubric:
    - Definition (0-2.5)
    - Diagram / Architecture description (0-2.5)
    - Procedural / Working steps (0-3.0)
    - Concrete Example (0-2.0)
    Returns detailed Senior feedback and points to improve for full marks.
    """
    if not _quiz_service:
        raise HTTPException(status_code=503, detail="Quiz service not initialized")
    
    result = _quiz_service.evaluate_answer(
        db=db,
        question_id=request.question_id,
        student_answer=request.student_answer,
        session_id=request.session_id
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# --- Section 10: Crowdsourced Upload Pipeline Endpoints ---

@router.post("/uploads", tags=["Crowdsourced Papers"], summary="Upload Student Past Paper PDF")
async def upload_past_paper(
    file: UploadFile = File(...),
    subject_code: Optional[str] = Form(None),
    year: Optional[int] = Form(None),
    session_term: Optional[str] = Form("Main"),
    db: Session = Depends(get_db)
):
    """
    Processes crowdsourced student PDF upload:
    - SHA-256 hash deduplication check
    - Header detection (Subject, Year, Branch)
    - Question boundary splitting
    - Automatic topic tagging via embedding similarity
    - Review queue placement
    """
    if not _upload_service:
        raise HTTPException(status_code=503, detail="Upload service not initialized")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    file_bytes = await file.read()
    if len(file_bytes) < 100:
        raise HTTPException(status_code=400, detail="Empty or corrupted PDF file.")

    result = _upload_service.process_pdf_upload(
        db=db,
        filename=file.filename,
        file_bytes=file_bytes,
        subject_id=subject_code,
        year=year
    )
    return result

@router.get("/uploads/{upload_id}", tags=["Crowdsourced Papers"], summary="Check Upload Processing Status")
def get_upload_status(upload_id: str, db: Session = Depends(get_db)):
    if not _upload_service:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    result = _upload_service.get_upload_status(db=db, upload_id=upload_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@router.get("/admin/review", tags=["Admin & Review Queue"], summary="List Papers in Review Queue")
def list_review_queue(db: Session = Depends(get_db)):
    if not _upload_service:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    return _upload_service.list_review_queue(db=db)

@router.post("/admin/review/{upload_id}/action", tags=["Admin & Review Queue"], summary="Approve or Reject Upload")
def review_upload_action(
    upload_id: str,
    action: str = Query(..., description="'approve' or 'reject'"),
    db: Session = Depends(get_db)
):
    if not _upload_service:
        raise HTTPException(status_code=503, detail="Upload service not initialized")
    result = _upload_service.approve_or_reject_upload(db=db, upload_id=upload_id, action=action)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

# --- Section 10: Feedback & Answer Correctness Tracking Endpoints ---

def _resolve_feedback_qa(db: Session, fb: Feedback) -> Dict[str, Any]:
    """Resolves the Question, Answer, and Subject linked to a Feedback record."""
    assistant_msg = (
        db.query(Message).filter(Message.id == fb.message_id).first()
        if fb.message_id else None
    )
    sess = (
        db.query(SessionModel).filter(SessionModel.id == fb.session_id).first()
        if fb.session_id else (assistant_msg.session if assistant_msg else None)
    )

    question_text = ""
    if assistant_msg and assistant_msg.session_id:
        user_msg = (
            db.query(Message)
            .filter(
                Message.session_id == assistant_msg.session_id,
                Message.role == "user",
                Message.created_at <= assistant_msg.created_at,
            )
            .order_by(Message.created_at.desc())
            .first()
        )
        if user_msg:
            question_text = user_msg.content
    elif fb.session_id:
        user_msg = (
            db.query(Message)
            .filter(Message.session_id == fb.session_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .first()
        )
        if user_msg:
            question_text = user_msg.content

    subj_short = None
    if sess and sess.subject_id:
        subj_rec = db.query(Subject).filter(Subject.id == sess.subject_id).first()
        subj_short = subj_rec.short_name if subj_rec else sess.subject_id.replace("SUBJ_", "")

    is_correct = fb.rating in ("up", "1", "true", "correct")
    return {
        "feedback_id": fb.id,
        "session_id": fb.session_id,
        "message_id": fb.message_id,
        "subject": subj_short or "General",
        "question": question_text or "(Question not recorded)",
        "answer": assistant_msg.content if assistant_msg else "",
        "rating": fb.rating,
        "is_correct": is_correct,
        "verdict": "Correct" if is_correct else "Incorrect",
        "comment": fb.comment,
        "created_at": fb.created_at.isoformat() if fb.created_at else None,
    }


@router.post("/feedback", tags=["Conversational RAG"], summary="Submit or Update Answer Correctness Feedback")
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    from datetime import datetime
    rating_str = str(request.rating).strip().lower()
    clean_rating = "up" if rating_str in ("up", "1", "true", "like", "correct") else "down"

    # Upsert if user flips thumbs-up/down or adds a reason on the same message_id
    existing = (
        db.query(Feedback).filter(Feedback.message_id == request.message_id).first()
        if request.message_id else None
    )
    if existing:
        existing.rating = clean_rating
        if request.comment is not None:
            existing.comment = request.comment
        if request.session_id and not existing.session_id:
            existing.session_id = request.session_id
        existing.created_at = datetime.utcnow()
        fb = existing
    else:
        fb = Feedback(
            id=f"fb_{uuid.uuid4().hex[:12]}",
            session_id=request.session_id,
            message_id=request.message_id,
            rating=clean_rating,
            comment=request.comment
        )
        db.add(fb)

    db.commit()

    # Synchronize feedback with LangSmith trace if available
    langsmith_target = request.trace_id or request.message_id
    if langsmith_target:
        send_feedback_to_langsmith(
            run_id=langsmith_target,
            rating=1 if clean_rating == "up" else -1,
            comment=request.comment
        )

    return {
        "status": "success",
        "feedback_id": fb.id,
        "rating": clean_rating,
        "is_correct": clean_rating == "up",
        "verdict": "Correct" if clean_rating == "up" else "Incorrect",
        "langsmith_synced": bool(langsmith_target)
    }


@router.get("/feedback", tags=["Conversational RAG"], summary="Track Answer Correctness & Feedback History")
def list_feedback_history(
    verdict: Optional[str] = Query(None, description="Filter by 'correct'/'up' or 'incorrect'/'down'"),
    subject: Optional[str] = Query(None, description="Optional subject short name filter e.g. 'DSA'"),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).all()
    resolved = [_resolve_feedback_qa(db, fb) for fb in rows]

    if subject and subject.strip().lower() != "all":
        subj_clean = subject.strip().upper()
        resolved = [r for r in resolved if (r["subject"] or "").upper() == subj_clean]

    total_rated = len(resolved)
    correct_count = sum(1 for r in resolved if r["is_correct"])
    incorrect_count = total_rated - correct_count
    accuracy_pct = round((correct_count / total_rated) * 100, 1) if total_rated else 0.0

    if verdict and verdict.strip().lower() != "all":
        v_low = verdict.strip().lower()
        want_correct = v_low in ("correct", "up", "1", "true")
        resolved = [r for r in resolved if r["is_correct"] == want_correct]

    return {
        "summary": {
            "total_rated": total_rated,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "accuracy_pct": accuracy_pct,
        },
        "records": resolved[:limit],
    }


@router.delete("/feedback/{feedback_id}", tags=["Conversational RAG"], summary="Delete a Feedback Entry")
def delete_feedback_entry(feedback_id: str, db: Session = Depends(get_db)):
    fb = db.query(Feedback).filter(Feedback.id == feedback_id).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Feedback entry not found")
    db.delete(fb)
    db.commit()
    return {"status": "deleted", "feedback_id": feedback_id}

# --- Section 10: Conversational RAG & Coaching Endpoints ---

@router.post("/chat", tags=["Conversational RAG"], summary="Multi-Turn Senior Tutor RAG Chat")
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """
    Executes a Senior Tutor conversational turn:
    - Contextual query rewriting
    - Hybrid retrieval (FAISS dense + BM25 sparse)
    - Cross-Encoder reranking
    - Senior Tutor persona (English / Hinglish)
    - 10-Mark / 2-Mark structured answer coaching with citations
    """
    if not _chat_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Chat service not initialized")

    filters_dict = request.filters.dict(exclude_none=True) if request.filters else {}

    try:
        response = _chat_service.process_turn(
            db=db,
            user_message=request.message,
            session_id=request.session_id or request.conversation_id,
            conversation_id=request.conversation_id,
            provider_name=request.provider,
            filters=filters_dict,
            top_k=request.top_k,
            language_pref=request.language,
            mode=request.mode,
            user_id=request.user_id,
            branch=request.branch or filters_dict.get("branch"),
            year=request.year or filters_dict.get("year")
        )
        return response
    except Exception as e:
        logger.error(f"Chat generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chat generation failed: {str(e)}")

# --- Direct Hybrid Search Endpoint ---

@router.post("/search", tags=["Search & Retrieval"], summary="Direct Hybrid Question Search")
def search(request: SearchRequest):
    """Direct hybrid semantic + keyword search with Cross-Encoder scoring without LLM generation."""
    if not _retrieval_pipeline or not _retrieval_pipeline._is_loaded:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Retrieval engine not loaded")

    results = _retrieval_pipeline.retrieve(
        query=request.query,
        subject=request.subject,
        semester=request.semester,
        year_from=request.year_from,
        year_to=request.year_to,
        section=request.section,
        marks=request.marks,
        top_k=request.top_k
    )
    return results

# --- Paper Archives & PDF Streaming Endpoints ---

@router.get("/papers", tags=["Paper Archives & PDF Streaming"], summary="Filter & Browse Examination Papers")
def list_papers(
    semester: Optional[int] = Query(None, description="Filter by semester 3, 4, 5"),
    subject: Optional[str] = Query(None, description="Filter by subject code or short name"),
    year: Optional[int] = Query(None, description="Filter by exam year"),
    db: Session = Depends(get_db)
):
    query = db.query(Paper)
    if semester:
        query = query.join(Subject, Subject.id == Paper.subject_id).join(SubjectOffering, SubjectOffering.subject_id == Subject.id).filter(
            SubjectOffering.semester == semester
        )
    if subject:
        if not semester:
            query = query.join(Subject, Subject.id == Paper.subject_id)
        query = query.filter(
            (Subject.short_name.ilike(f"%{subject}%")) | (Subject.code.ilike(f"%{subject}%"))
        )
    if year:
        query = query.filter(Paper.year == year)

    papers = query.distinct().order_by(Paper.year.desc()).all()
    results = []
    for p in papers:
        subj = p.subject
        sem_val = (subj.offerings[0].semester if subj and subj.offerings else 3)
        results.append({
            "paper_id": p.id,
            "subject": subj.name if subj else "Engineering Subject",
            "subject_code": subj.code if subj else "RTU",
            "subject_short": subj.short_name if subj else "SUBJ",
            "semester": sem_val,
            "year": p.year,
            "session": p.session,
            "page_count": p.page_count,
            "file_size_kb": p.file_size_kb,
            "source_pdf": p.source_pdf,
            "relative_path": p.relative_path,
            "question_count": len(p.questions)
        })
    return results

@router.get("/papers/{paper_id}", tags=["Paper Archives & PDF Streaming"], summary="Get Paper Details & Questions")
def get_paper(paper_id: str, db: Session = Depends(get_db)):
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    subj = paper.subject
    return {
        "paper_id": paper.id,
        "subject": subj.name if subj else "Engineering Subject",
        "subject_code": subj.code if subj else "RTU",
        "subject_short": subj.short_name if subj else "SUBJ",
        "year": paper.year,
        "session": paper.session,
        "page_count": paper.page_count,
        "file_size_kb": paper.file_size_kb,
        "source_pdf": paper.source_pdf,
        "questions": [
            {
                "question_id": q.id,
                "section": q.part,
                "marks": q.marks,
                "question_text": q.text,
                "page_number": q.page_number,
                "topic_name": q.topic.name if q.topic else None
            }
            for q in sorted(paper.questions, key=lambda x: (x.part or "", x.id))
        ]
    }

@router.get("/papers/{paper_id}/pdf", tags=["Paper Archives & PDF Streaming"], summary="Stream Original Scanned PDF")
def serve_paper_pdf(paper_id: str, db: Session = Depends(get_db)):
    """Streams physical PDF binary for direct browser embedding and page jumping."""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    if not paper:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Paper not found")

    rel = paper.relative_path or ""
    candidates = [
        settings.WORKSPACE_ROOT / rel,
        settings.WORKSPACE_ROOT / "cleaned_dataset" / rel,
        settings.WORKSPACE_ROOT / "cleaned_dataset" / (paper.source_pdf or ""),
        settings.STORAGE_DIR / "student_uploads" / (paper.source_pdf or ""),
    ]
    pdf_path = next((c for c in candidates if rel and c.exists() and c.is_file()), None)
    if pdf_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Physical PDF file not found for {paper.source_pdf}"
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=paper.source_pdf
    )

# --- Curriculum & Academic Analytics Endpoints ---

@router.get("/subjects", tags=["Curriculum & Analytics"], summary="List Curriculum Subjects with Data Tiers")
def list_subjects(
    branch: Optional[str] = Query(None, description="Filter by branch e.g. 'CSE', 'IT'"),
    semester: Optional[int] = Query(None, description="Filter by semester 1-8"),
    year: Optional[int] = Query(None, description="Filter by year 1-4 (Year 1: Sem 1-2, Year 2: Sem 3-4, Year 3: Sem 5-6, Year 4: Sem 7-8)"),
    db: Session = Depends(get_db)
):
    query = db.query(Subject)
    subjects = query.all()

    results = []
    for s in subjects:
        offerings = s.offerings
        sem_list = [o.semester for o in offerings]
        branch_list = [o.branch.code for o in offerings if o.branch]
        
        # Determine year(s) for this subject (Sem 1-2: Year 1, Sem 3-4: Year 2, Sem 5-6: Year 3, Sem 7-8: Year 4)
        year_list = list(set((sem + 1) // 2 for sem in sem_list))
        primary_year = min(year_list) if year_list else 2

        # Check branch / semester / year filtering
        if branch and not any(o.branch.code.upper() == branch.upper() for o in offerings if o.branch):
            continue
        if semester and not any(o.semester == semester for o in offerings):
            continue
        if year and year not in year_list:
            continue

        paper_cnt = len(s.papers)

        results.append({
            "subject_id": s.id,
            "name": s.name,
            "subject_code": s.code,
            "subject_short": s.short_name,
            "short_name": s.short_name,
            "data_tier": s.data_tier or ("Tier A" if paper_cnt >= 5 else ("Tier B" if paper_cnt >= 1 else "Tier C")),
            "semesters": sem_list,
            "branches": list(set(branch_list)),
            "year": primary_year,
            "years": year_list,
            "paper_count": paper_cnt,
            "question_count": sum(len(p.questions) for p in s.papers),
            "unit_count": len(s.units)
        })

    results.sort(key=lambda x: (x["year"], x["data_tier"], x["subject_short"]))
    return results

@router.get("/analytics/summary", tags=["Curriculum & Analytics"], summary="Overall Corpus Analytics")
def get_analytics_summary(db: Session = Depends(get_db)):
    if not _analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    return _analytics_service.get_summary(db=db)

@router.get("/analytics/repeated", tags=["Curriculum & Analytics"], summary="Discover Repeated Exam Questions")
def get_repeated_questions(
    subject: Optional[str] = Query(None, description="Subject short code e.g. 'DSA'"),
    threshold: float = Query(0.75, ge=0.5, le=1.0),
    limit: int = Query(25, ge=1, le=100),
    db: Session = Depends(get_db)
):
    if not _analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    return _analytics_service.find_repeated_questions(
        db=db,
        subject_short=subject,
        threshold=threshold,
        limit=limit
    )

# --- Observability, Screening & LangSmith Persistence Endpoints ---

@router.get("/observability/status", tags=["Observability & Screening"], summary="LangSmith & Screening Guardrails Status")
def get_observability_status():
    """Returns real-time status of LangSmith tracing, active project, and academic screening guardrails."""
    return {
        "status": "healthy",
        "langsmith_tracing_active": is_tracing_active(),
        "langsmith_project": get_current_project_name(),
        "screening_guardrails_enabled": True,
        "input_screeners": [
            "academic_scope_screener",
            "prompt_injection_guardrail",
            "syllabus_boundary_check"
        ],
        "output_screeners": [
            "anti_leakage_sanitizer",
            "format_rubric_evaluator",
            "groundedness_checker"
        ],
        "evaluators": [
            "syllabus_adherence",
            "format_compliance",
            "citation_groundedness"
        ]
    }

@router.post("/observability/sync-dataset", tags=["Observability & Screening"], summary="Sync RTU Benchmark Dataset to LangSmith Cloud")
def sync_dataset_to_langsmith():
    """Persists the 300-question RTU examination benchmark dataset into LangSmith Cloud."""
    return sync_benchmark_to_langsmith()


@router.get("/analytics/compare", tags=["Curriculum & Analytics"], summary="Compare Questions Across Two Exam Years")
def compare_exam_years(
    subject: str = Query(..., description="Subject code e.g. 'DSA'"),
    year_a: int = Query(..., ge=2000, le=2030),
    year_b: int = Query(..., ge=2000, le=2030),
    db: Session = Depends(get_db)
):
    if not _analytics_service:
        raise HTTPException(status_code=503, detail="Analytics service not initialized")
    return _analytics_service.compare_years(
        db=db,
        subject_short=subject,
        year_a=year_a,
        year_b=year_b
    )
