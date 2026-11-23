"""
LangChain Hybrid Retriever for RTU AI Academic Assistant
Combines dense FAISS search, sparse BM25 search via EnsembleRetriever (weighted RRF),
and contextual reranking via CrossEncoderReranker.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker

from app.core.config import settings
from app.retrieval.document_preparer import prepare_retrieval_documents
from app.database.session import SessionLocal

_CACHED_RETRIEVER = None


def load_questions_from_db(db=None) -> List[Dict[str, Any]]:
    """Loads prepared question documents from SQLite database."""
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True
    try:
        return prepare_retrieval_documents(db)
    finally:
        if close_db:
            db.close()


def build_retriever(rows: Optional[List[Dict[str, Any]]] = None, db=None, force_rebuild: bool = False):
    """
    Builds or loads the LangChain ContextualCompressionRetriever composed of:
    1. Dense Retriever: FAISS with BAAI/bge-small-en-v1.5 embeddings (cached to disk)
    2. Sparse Retriever: BM25Retriever over canonical questions
    3. Hybrid Ensemble: EnsembleRetriever with Reciprocal Rank Fusion (weights=[0.5, 0.5])
    4. Reranker Compressor: CrossEncoderReranker (cross-encoder/ms-marco-MiniLM-L-6-v2, top_n=5)
    """
    global _CACHED_RETRIEVER
    if _CACHED_RETRIEVER is not None and not force_rebuild and rows is None:
        return _CACHED_RETRIEVER

    if rows is None:
        rows = load_questions_from_db(db)

    # Convert SQLite rows into LangChain Document objects preserving citation metadata
    docs = [
        Document(
            page_content=r.get("question_text", ""),
            metadata={
                "question_id": r.get("question_id", ""),
                "subject": r.get("subject_short") or r.get("subject_code") or r.get("subject") or "RTU",
                "subject_code": r.get("subject_code", ""),
                "subject_name": r.get("subject", ""),
                "year": r.get("year", 2022),
                "section": r.get("section") or r.get("part") or "Part B",
                "marks": r.get("marks", 10),
                "source_pdf": r.get("source_pdf", "RTU_Exam.pdf"),
                "page_number": r.get("page_number", 1),
                "topic_name": r.get("topic_name", ""),
                "semester": r.get("semester", 3),
            }
        )
        for r in rows
        if r.get("question_text", "").strip()
    ]

    # Initialize Embedding Model
    emb = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL)

    # Persistent FAISS vector store
    lc_faiss_dir = settings.INDEXES_DIR / "lc_faiss"
    faiss_index_file = lc_faiss_dir / "index.faiss"

    if faiss_index_file.exists() and not force_rebuild:
        dense_store = FAISS.load_local(
            str(lc_faiss_dir),
            emb,
            allow_dangerous_deserialization=True
        )
    else:
        lc_faiss_dir.mkdir(parents=True, exist_ok=True)
        dense_store = FAISS.from_documents(docs, emb)
        dense_store.save_local(str(lc_faiss_dir))

    dense = dense_store.as_retriever(search_kwargs={"k": settings.HYBRID_CANDIDATE_K})

    # BM25 Sparse Keyword Retriever
    sparse = BM25Retriever.from_documents(docs)
    sparse.k = settings.HYBRID_CANDIDATE_K

    # Hybrid Ensemble (Reciprocal Rank Fusion)
    hybrid = EnsembleRetriever(
        retrievers=[dense, sparse],
        weights=[0.5, 0.5]
    )

    # Cross-Encoder Reranker Compressor
    cross_encoder = HuggingFaceCrossEncoder(model_name=settings.RERANKER_MODEL)
    reranker = CrossEncoderReranker(
        model=cross_encoder,
        top_n=settings.FINAL_TOP_K
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=hybrid
    )

    _CACHED_RETRIEVER = compression_retriever
    return compression_retriever
