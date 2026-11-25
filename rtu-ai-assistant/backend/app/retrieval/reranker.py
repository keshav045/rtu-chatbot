"""
Cross-Encoder Reranker using cross-encoder/ms-marco-MiniLM-L-6-v2
Scores candidate question passages directly against the student query.
Includes safe memory fallback for low-RAM cloud containers (e.g. Render 512MB).

v2 additions:
- MMR (Maximal Marginal Relevance) post-processing to diversify final top_k results.
  Uses Jaccard similarity on question text tokens to measure redundancy.
  Lambda=0.7 (70% relevance, 30% diversity) prevents near-duplicate questions.
"""

import os
import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger("rtu.retrieval")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# TASK 3: MMR parameters
MMR_LAMBDA = 0.7  # weight for relevance vs. diversity


def _tokenize(text: str) -> Set[str]:
    """Simple whitespace+lowercase tokenizer for Jaccard similarity."""
    return set(text.lower().split())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _mmr_select(
    candidates: List[Dict[str, Any]],
    top_k: int,
    mmr_lambda: float = MMR_LAMBDA,
) -> List[Dict[str, Any]]:
    """
    Maximal Marginal Relevance selection.

    Parameters
    ----------
    candidates : list of dicts, already sorted descending by rerank_score.
    top_k      : number of items to select.
    mmr_lambda : trade-off between relevance (1.0) and diversity (0.0).

    Returns
    -------
    A re-ordered list of up to top_k items that balance relevance and diversity.
    """
    if len(candidates) <= top_k:
        return candidates  # nothing to trim / diversify

    # Pre-tokenise every candidate text once
    token_sets = [_tokenize(c.get("document_text", "")) for c in candidates]

    selected_indices: List[int] = []
    remaining_indices = list(range(len(candidates)))

    while len(selected_indices) < top_k and remaining_indices:
        if not selected_indices:
            # Seed with the highest-relevance doc (index 0, already sorted)
            best_idx = remaining_indices[0]
        else:
            best_score = float("-inf")
            best_idx = remaining_indices[0]

            for i in remaining_indices:
                # Relevance component: normalised rerank_score (use rank-based proxy)
                rel_score = candidates[i].get("rerank_score", 0.0)

                # Diversity component: max Jaccard with any already-selected doc
                max_sim = max(
                    _jaccard(token_sets[i], token_sets[j])
                    for j in selected_indices
                )

                mmr_score = mmr_lambda * rel_score - (1.0 - mmr_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return [candidates[i] for i in selected_indices]


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
                self.model = CrossEncoder(RERANKER_MODEL, device="cpu")
                logger.info("CrossEncoder reranker loaded on CPU successfully.")
            except Exception as e:
                logger.warning(f"CrossEncoder reranker skipped in low-memory environment: {e}")
                self.model = None

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Reranks candidates using cross-encoder cross-attention, then applies MMR to
        the final top_k to prevent near-duplicate results.
        Gracefully falls back to existing RRF ranking if memory is constrained.
        """
        if not candidates:
            return []

        self._load_model()
        if self.model is None:
            # Safe fallback: keep RRF order without inflating logit scores above out-of-scope gates
            for i, doc in enumerate(candidates):
                doc = dict(doc)
                doc["rerank_fallback"] = True
                if "rerank_score" not in doc:
                    doc["rerank_score"] = round(-7.5 - 0.1 * i, 4)
                candidates[i] = doc
            return _mmr_select(candidates[:top_k * 2], top_k)

        try:
            pairs = [(query, c["document_text"]) for c in candidates]
            scores = self.model.predict(pairs)

            reranked = []
            for doc, score in zip(candidates, scores):
                doc_copy = dict(doc)
                doc_copy["rerank_score"] = float(score)
                reranked.append(doc_copy)

            reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

            # TASK 3: Apply MMR to diversify the top_k final results
            diversified = _mmr_select(reranked, top_k)
            return diversified

        except Exception as e:
            logger.warning(f"Reranking encountered error, falling back to RRF: {e}")
            return candidates[:top_k]

# Backward-compatible alias
Reranker = CrossEncoderReranker
