"""
Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2
Scores candidate question passages directly against the student query.
Includes safe memory fallback for low-RAM cloud containers (e.g. Render 512MB).
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger("rtu.retrieval")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    def __init__(self):
        self.model = None
        # In memory-constrained environments, reranker can be disabled via environment variable
        self._disabled = os.getenv("DISABLE_RERANKER", "false").lower() == "true"

    def _load_model(self):
        if self._disabled:
            return
        if self.model is None:
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass
            try:
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(RERANKER_MODEL)
                logger.info("CrossEncoder reranker loaded successfully.")
            except Exception as e:
                logger.warning(f"CrossEncoder reranker skipped in low-memory environment: {e}")
                self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Reranks candidates using cross-encoder cross-attention.
        Gracefully falls back to existing RRF ranking if memory is constrained.
        """
        if not candidates:
            return []

        self._load_model()
        if self.model is None:
            # High-accuracy fallback: candidates are already ranked by Reciprocal Rank Fusion
            return candidates[:top_k]

        try:
            pairs = [(query, c["document_text"]) for c in candidates]
            scores = self.model.predict(pairs)

            reranked = []
            for doc, score in zip(candidates, scores):
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = float(score)
                reranked.append(doc_copy)

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception as e:
            logger.warning(f"Reranking encountered error, falling back to RRF: {e}")
            return candidates[:top_k]
