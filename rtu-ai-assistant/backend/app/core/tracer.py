"""
Comprehensive LangSmith Observability & Tracing Engine for RTU AI Academic Assistant
Provides:
1. Automatic dual-mode environment configuration (LANGCHAIN_* and LANGSMITH_*).
2. LangSmith Client singleton for feedback logging, run retrieval, and dataset management.
3. User Feedback synchronization (thumbs up/down directly posted to LangSmith runs).
4. Screening Evaluator synchronization (scope, safety, format, and groundedness metrics).
5. Safe fallback decorators if LangSmith is unconfigured or unreachable.
"""

import os
import uuid
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("rtu.tracer")

# Global LangSmith client cache
_LANGSMITH_CLIENT = None
_IS_TRACING_ENABLED = False


def setup_langsmith_tracing(project_name: str = "rtu-ai-academic-assistant") -> bool:
    """
    Initializes LangSmith tracing if LANGCHAIN_API_KEY or LANGSMITH_API_KEY is present.
    Sets both LANGCHAIN_* and LANGSMITH_* environment variables for 100% compatibility across LangChain versions.
    """
    global _IS_TRACING_ENABLED, _LANGSMITH_CLIENT

    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.info("LangSmith API key not configured; tracing is running in local-telemetry mode.")
        _IS_TRACING_ENABLED = False
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGSMITH_API_KEY"] = api_key

    project = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or project_name
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ["LANGSMITH_PROJECT"] = project

    endpoint = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGSMITH_ENDPOINT"] = endpoint

    _IS_TRACING_ENABLED = True
    logger.info(f"LangSmith Tracing successfully configured for project: '{project}' at '{endpoint}'.")
    return True


def get_langsmith_client():
    """Returns the cached LangSmith Client or initializes one if credentials exist."""
    global _LANGSMITH_CLIENT
    if _LANGSMITH_CLIENT is not None:
        return _LANGSMITH_CLIENT

    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        return None

    try:
        from langsmith import Client
        _LANGSMITH_CLIENT = Client(
            api_key=api_key,
            api_url=os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"
        )
        return _LANGSMITH_CLIENT
    except Exception as e:
        logger.warning(f"Could not initialize LangSmith Client: {e}")
        return None


def is_tracing_active() -> bool:
    """Returns True if LangSmith API credentials and tracing are actively enabled."""
    return bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))


def get_current_project_name() -> str:
    """Returns active LangSmith project name."""
    return os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "rtu-ai-academic-assistant"


def send_feedback_to_langsmith(
    run_id: str,
    rating: int,
    comment: Optional[str] = None
) -> bool:
    """
    Submits user feedback (thumbs up +1, thumbs down -1) to LangSmith for the specified run ID.
    Enables tracking of student satisfaction curves and diagnostic filtering in the LangSmith dashboard.
    """
    if not run_id:
        return False

    client = get_langsmith_client()
    if not client:
        logger.debug("LangSmith client unavailable; feedback saved to local database only.")
        return False

    try:
        # Standard score normalized: 1.0 (thumbs up), 0.0 (thumbs down)
        norm_score = 1.0 if rating > 0 else 0.0
        client.create_feedback(
            run_id=run_id,
            key="user_score",
            score=norm_score,
            comment=comment or ("Helpful exam explanation" if rating > 0 else "Needs improvement"),
            value="positive" if rating > 0 else "negative"
        )
        logger.info(f"Successfully recorded user feedback in LangSmith for run: {run_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to record LangSmith feedback for run {run_id}: {e}")
        return False


def send_screening_eval_to_langsmith(
    run_id: str,
    screening_scores: Dict[str, float]
) -> bool:
    """
    Persists automated screening evaluation metrics to the LangSmith run:
    - scope_score (0.0 - 1.0)
    - safety_score (0.0 - 1.0)
    - format_score (0.0 - 1.0)
    - groundedness_score (0.0 - 1.0)
    """
    if not run_id or not screening_scores:
        return False

    client = get_langsmith_client()
    if not client:
        return False

    try:
        for metric_name, score_val in screening_scores.items():
            client.create_feedback(
                run_id=run_id,
                key=metric_name,
                score=float(score_val),
                comment="Automated RTU Academic Screening Guardrail"
            )
        return True
    except Exception as e:
        logger.warning(f"Failed to record screening evaluation in LangSmith for run {run_id}: {e}")
        return False


def generate_trace_id() -> str:
    """Generates a UUID4 trace identifier."""
    return str(uuid.uuid4())
