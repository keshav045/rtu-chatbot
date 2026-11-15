"""
Application Configuration and Settings for RTU AI Academic Assistant
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict

# Resolve directories
APP_DIR = Path(__file__).resolve().parent.parent           # rtu-ai-assistant/backend/app
BACKEND_DIR = APP_DIR.parent                              # rtu-ai-assistant/backend
ASSISTANT_ROOT = BACKEND_DIR.parent                       # rtu-ai-assistant
WORKSPACE_ROOT = ASSISTANT_ROOT.parent                    # rtu project

class Settings(BaseSettings):
    PROJECT_NAME: str = "RTU AI Academic Assistant"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Paths
    WORKSPACE_ROOT: Path = WORKSPACE_ROOT
    PROJECT_ROOT: Path = WORKSPACE_ROOT
    ASSISTANT_ROOT: Path = ASSISTANT_ROOT
    BACKEND_DIR: Path = BACKEND_DIR
    STORAGE_DIR: Path = BACKEND_DIR / "data" / "storage"
    INDEXES_DIR: Path = BACKEND_DIR / "data" / "indexes"
    CLEANED_DATASET_DIR: Path = WORKSPACE_ROOT / "cleaned_dataset" / "question_papers"

    # Database
    SQLITE_DB_PATH: Path = STORAGE_DIR / "rtu_metadata.db"
    DATABASE_URL: str = f"sqlite:///{SQLITE_DB_PATH.as_posix()}"

    # Retrieval Models & Config
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    FAISS_INDEX_PATH: Path = INDEXES_DIR / "rtu_faiss_bge.index"
    FAISS_MAPPING_PATH: Path = INDEXES_DIR / "faiss_id_mapping.json"
    BM25_INDEX_PATH: Path = INDEXES_DIR / "bm25_index.pkl"

    # Search defaults
    HYBRID_CANDIDATE_K: int = 25
    FINAL_TOP_K: int = 5
    RRF_K: int = 60
    OUT_OF_SCOPE_THRESHOLD: float = -9.5

    # LLM Settings
    DEFAULT_LLM_PROVIDER: str = "ollama"  # 'ollama', 'groq', 'gemini', 'openai', 'offline'
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "rtu-senior"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    OPENAI_MODEL: str = "gpt-4o-mini"

    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*"
    ]

    model_config = ConfigDict(env_file=".env", extra="allow")

settings = Settings()

# Ensure directories exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEXES_DIR.mkdir(parents=True, exist_ok=True)
