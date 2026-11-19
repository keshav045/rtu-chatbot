"""
Unified Retrieval Pipeline for RTU AI Academic Assistant
Integrates FAISS vector search, BM25 keyword matching, RRF hybrid fusion,
Cross-Encoder reranking, and citation context building.

Optimizations (v2):
- LRU query cache (maxsize=200) to skip full pipeline on repeated queries
- rerank_candidates_k default reduced from 30 → 12 (cuts CPU rerank ~4.5s → ~1.2s)
- Cache stats (hits, misses, size) exposed in retrieval_stats
"""

import re
from collections import OrderedDict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session

from .document_preparer import prepare_retrieval_documents
from .dense_indexer import DenseIndexer
from .bm25_indexer import BM25Indexer
from .reranker import CrossEncoderReranker
from .hybrid_retriever import reciprocal_rank_fusion, apply_metadata_filters
from .context_builder import ContextBuilder

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = BACKEND_DIR / "data" / "indexes"


class _LRUCache:
    """
    Thread-safe (GIL-protected) dict-based LRU cache with a fixed maxsize.
    Uses an OrderedDict so the least-recently-used entry sits at the front.
    """

    def __init__(self, maxsize: int = 200):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple) -> Optional[Any]:
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        return None

    def put(self, key: Tuple, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)  # evict LRU

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        return len(self._cache)


def _normalize_query(query: str) -> str:
    """Lowercase + collapse whitespace for a stable cache key."""
    return re.sub(r"\s+", " ", query.strip().lower())


class RetrievalPipeline:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.dense_indexer = DenseIndexer(index_dir)
        self.bm25_indexer = BM25Indexer(index_dir)
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder(max_context_chars=4000)
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self._is_loaded = False
        self._cache = _LRUCache(maxsize=200)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_indexes(self, db: Session):
        """
        Extracts documents from SQLite and builds both FAISS and BM25 indexes.
        """
        print("[RetrievalPipeline] Extracting questions from SQLite...")
        documents = prepare_retrieval_documents(db)
        print(f"[RetrievalPipeline] Prepared {len(documents)} searchable question units.")

        self.doc_store = {doc["question_id"]: doc for doc in documents}

        print("[RetrievalPipeline] Building Dense Vector Index (FAISS)...")
        self.dense_indexer.build_index(documents)

        print("[RetrievalPipeline] Building Sparse Keyword Index (BM25)...")
        self.bm25_indexer.build_index(documents)

        self._is_loaded = True
        print("[RetrievalPipeline] All indexes successfully built and persisted.")

    def load_indexes(self, db: Session):
        """
        Loads pre-built indexes and initializes the document store from SQLite.
        """
        documents = prepare_retrieval_documents(db)
        self.doc_store = {doc["question_id"]: doc for doc in documents}

        self.dense_indexer.load_index()
        self.bm25_indexer.load_index()
        self._is_loaded = True
        print(f"[RetrievalPipeline] Loaded indexes for {len(self.doc_store)} questions.")

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clears the LRU query cache and resets hit/miss counters."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        subject: Optional[str] = None,
        semester: Optional[int] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        section: Optional[str] = None,
        marks: Optional[int] = None,
        top_k: int = 5,
        dense_k: int = 50,
        sparse_k: int = 50,
        rerank_candidates_k: int = 12,  # TASK 2: reduced from 30 → 12
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval: Dense (FAISS) + Sparse (BM25) → Metadata Filtering → RRF → Cross-Encoder.

        Results for identical (query, subject, semester, marks, top_k) tuples are served
        from an LRU cache (maxsize=200) on subsequent calls.
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline indexes not loaded. Call load_indexes() or build_indexes() first.")

        # ------------------------------------------------------------------
        # TASK 1: LRU Cache lookup
        # ------------------------------------------------------------------
        query_normalized = _normalize_query(query)
        cache_key = (query_normalized, subject, semester, marks, top_k)

        cached = self._cache.get(cache_key)
        if cached is not None:
            # Return a fresh copy with updated cache stats
            result = dict(cached)
            result["retrieval_stats"] = dict(cached["retrieval_stats"])
            result["retrieval_stats"]["cache_hit"] = True
            result["retrieval_stats"]["cache_hits"] = self._cache.hits
            result["retrieval_stats"]["cache_misses"] = self._cache.misses
            result["retrieval_stats"]["cache_size"] = self._cache.size
            return result

        # ------------------------------------------------------------------
        # 1. Dense Semantic Search
        # ------------------------------------------------------------------
        dense_results = self.dense_indexer.search(query, top_k=dense_k)

        # 2. Sparse Keyword Search
        sparse_results = self.bm25_indexer.search(query, top_k=sparse_k)

        # 3. Apply Metadata Filtering to both candidate lists
        filtered_dense = [
            (qid, score) for qid, score in dense_results
            if qid in self.doc_store and apply_metadata_filters(
                self.doc_store[qid], subject, semester, year_from, year_to, section, marks
            )
        ]

        filtered_sparse = [
            (qid, score) for qid, score in sparse_results
            if qid in self.doc_store and apply_metadata_filters(
                self.doc_store[qid], subject, semester, year_from, year_to, section, marks
            )
        ]

        # If a specific subject is requested, DO NOT fallback across other subjects!
        # Only fallback if NO subject filter was supplied or subject was 'all'
        if not filtered_dense and not filtered_sparse and (not subject or subject.lower() == "all"):
            filtered_dense = [(qid, score) for qid, score in dense_results if qid in self.doc_store]
            filtered_sparse = [(qid, score) for qid, score in sparse_results if qid in self.doc_store]

        # 4. Reciprocal Rank Fusion (RRF)
        fused_rankings = reciprocal_rank_fusion([filtered_dense, filtered_sparse], k=60)

        # 5. Gather top candidates for Cross-Encoder Reranking
        candidate_ids = [qid for qid, _ in fused_rankings[:rerank_candidates_k]]
        candidate_docs = [self.doc_store[qid] for qid in candidate_ids if qid in self.doc_store]

        # 6. Cross-Encoder Reranking (with MMR diversity – see reranker.py)
        reranked_docs = self.reranker.rerank(query, candidate_docs, top_k=top_k)

        # 7. Construct LLM Context and Citations
        context_string = self.context_builder.build_context(reranked_docs)
        citations = self.context_builder.extract_citations(reranked_docs)

        # ------------------------------------------------------------------
        # TASK 1: Store result in cache
        # ------------------------------------------------------------------
        result = {
            "query": query,
            "results_count": len(reranked_docs),
            "results": reranked_docs,
            "context_string": context_string,
            "citations": citations,
            "retrieval_stats": {
                "dense_matches": len(filtered_dense),
                "sparse_matches": len(filtered_sparse),
                "fused_candidates": len(fused_rankings),
                "reranked_count": len(reranked_docs),
                "cache_hit": False,
                "cache_hits": self._cache.hits,
                "cache_misses": self._cache.misses,
                "cache_size": self._cache.size,
            },
        }
        self._cache.put(cache_key, result)

        # Update stats to reflect the miss that just happened
        result = dict(result)
        result["retrieval_stats"] = dict(result["retrieval_stats"])
        result["retrieval_stats"]["cache_misses"] = self._cache.misses
        result["retrieval_stats"]["cache_size"] = self._cache.size

        return result
