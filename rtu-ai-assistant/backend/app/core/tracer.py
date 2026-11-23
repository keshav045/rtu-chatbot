"""
LangSmith Tracing Configuration for RTU AI Academic Assistant
Automatically instruments LangChain runs, LCEL chains, and custom spans when enabled.
"""

import os
import logging

logger = logging.getLogger("rtu.tracer")


def setup_langsmith_tracing(project_name: str = "rtu-ai-academic-assistant") -> bool:
    """
    Initializes LangSmith tracing if LANGCHAIN_API_KEY or LANGSMITH_API_KEY is present.
    Sets both LANGCHAIN_* and LANGSMITH_* environment variables for 100% compatibility across LangChain versions.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.debug("LangSmith API key not configured; tracing is inactive.")
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

    logger.info(f"LangSmith Tracing enabled for project: {project}")
    return True
