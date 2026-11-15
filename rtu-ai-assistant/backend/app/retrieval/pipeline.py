"""
Unified Retrieval Pipeline for RTU AI Academic Assistant
Integrates FAISS vector search, BM25 keyword matching, RRF hybrid fusion,
Cross-Encoder reranking, and citation context building.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from .document_preparer import prepare_retrieval_documents
from .dense_indexer import DenseIndexer
from .bm25_indexer import BM25Indexer
from .reranker import CrossEncoderReranker
from .hybrid_retriever import reciprocal_rank_fusion, apply_metadata_filters
from .context_builder import ContextBuilder

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_DIR = BACKEND_DIR / "data" / "indexes"

class RetrievalPipeline:
    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.dense_indexer = DenseIndexer(index_dir)
        self.bm25_indexer = BM25Indexer(index_dir)
        self.reranker = CrossEncoderReranker()
        self.context_builder = ContextBuilder(max_context_chars=4000)
        self.doc_store: Dict[str, Dict[str, Any]] = {}
        self._is_loaded = False

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
        rerank_candidates_k: int = 30
    ) -> Dict[str, Any]:
        """
        Executes hybrid retrieval: Dense (FAISS) + Sparse (BM25) -> Metadata Filtering -> RRF -> Cross-Encoder.
        """
        if not self._is_loaded:
            raise RuntimeError("Pipeline indexes not loaded. Call load_indexes() or build_indexes() first.")

        # 1. Dense Semantic Search
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

        # Graceful fallback: If the strict subject filter yielded 0 candidates, search across all subjects
        # (e.g. student is in DSA session but asks an OS, DBMS, or general engineering question)
        if not filtered_dense and not filtered_sparse:
            filtered_dense = [(qid, score) for qid, score in dense_results if qid in self.doc_store]
            filtered_sparse = [(qid, score) for qid, score in sparse_results if qid in self.doc_store]

        # 4. Reciprocal Rank Fusion (RRF)
        fused_rankings = reciprocal_rank_fusion([filtered_dense, filtered_sparse], k=60)

        # 5. Gather top candidates for Cross-Encoder Reranking
        candidate_ids = [qid for qid, _ in fused_rankings[:rerank_candidates_k]]
        candidate_docs = [self.doc_store[qid] for qid in candidate_ids if qid in self.doc_store]

        # 6. Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank(query, candidate_docs, top_k=top_k)

        # 7. Construct LLM Context and Citations
        context_string = self.context_builder.build_context(reranked_docs)
        citations = self.context_builder.extract_citations(reranked_docs)

        return {
            "query": query,
            "results_count": len(reranked_docs),
            "results": reranked_docs,
            "context_string": context_string,
            "citations": citations,
            "retrieval_stats": {
                "dense_matches": len(filtered_dense),
                "sparse_matches": len(filtered_sparse),
                "fused_candidates": len(fused_rankings),
                "reranked_count": len(reranked_docs)
            }
        }

