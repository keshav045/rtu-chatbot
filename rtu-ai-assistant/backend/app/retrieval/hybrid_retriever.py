"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF) and Metadata Filtering
Combines dense vector retrieval (FAISS) and sparse keyword retrieval (BM25).
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

def reciprocal_rank_fusion(rankings: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Combines multiple ranked lists of (doc_id, score) into a single fused ranking
    using standard Reciprocal Rank Fusion (RRF).
    """
    rrf_scores = defaultdict(float)

    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_results

def apply_metadata_filters(
    doc: Dict[str, Any],
    subject: Optional[str] = None,
    semester: Optional[int] = None,
    year_from: Optional[int] = None,
    year_to: Optional[int] = None,
    section: Optional[str] = None,
    marks: Optional[int] = None
) -> bool:
    """
    Returns True if the document satisfies all supplied metadata filter constraints.
    """
    if subject and subject.lower() != "all":
        subj_clean = subject.replace("SUBJ_", "").strip().lower()
        sub_match = (
            doc["subject_short"].lower() == subject.lower() or
            doc["subject_short"].lower() == subj_clean or
            doc["subject"].lower() == subject.lower() or
            doc["subject_code"].lower() == subject.lower() or
            doc.get("subject_id", "").lower() == subject.lower()
        )
        if not sub_match:
            return False

    if semester and semester != 0:
        if int(doc.get("semester", 0)) != int(semester):
            return False

    if year_from is not None:
        if int(doc.get("year", 0)) < int(year_from):
            return False

    if year_to is not None:
        if int(doc.get("year", 0)) > int(year_to):
            return False

    if section and section.lower() != "all":
        sec_str = str(doc.get("section", "")).lower()
        if section.lower() not in sec_str:
            return False

    if marks and marks > 0:
        if int(doc.get("marks", 0)) != int(marks):
            return False

    return True

