"""
RTU Syllabus Topic & Past-Paper Verification and Related-Topic Retrieval Engine.

1. Indexes all RTU syllabus topics (`topics` JOIN `units` JOIN `subjects` + `aliases`)
   using the shared BGE embedding model (`BAAI/bge-small-en-v1.5`) with persistent NPZ caching.
2. Tests any user query against BOTH RTU syllabus topics and RTU past-paper questions
   before generation to classify RTU-related vs out-of-the-box / out-of-syllabus queries
   (even for subjects with 0 extracted past-paper questions).
3. When verified as RTU-related, retrieves ALL closely related syllabus topics, unit
   context, and topic-linked past-paper questions for full grounded context.
"""

import difflib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.models import Paper, Question, Subject, SubjectOffering, Topic, Unit
from app.retrieval.document_preparer import is_meaningful_question

logger = logging.getLogger("rtu.topic_verifier")

_SHORT_VALID_TOKENS = {
    "os", "ai", "ml", "dl", "ds", "db", "er", "io", "ip", "cn", "de", "se",
    "cd", "cg", "ca", "cc", "bc", "qa", "ll", "lr", "np", "p", "c", "o",
}
_STOPWORDS = {
    "what", "when", "where", "which", "whom", "whose", "why", "how", "explain",
    "define", "describe", "discuss", "compare", "differentiate", "write", "give",
    "tell", "make", "create", "list", "show", "find", "with", "from", "into",
    "about", "that", "this", "these", "those", "their", "there", "your", "have",
    "will", "would", "could", "should", "mark", "marks", "part", "unit", "question",
    "questions", "answer", "answers", "example", "examples", "steps", "algorithm",
    "concept", "topic", "topics", "between", "using", "short", "notes", "note",
    "detail", "details", "system", "systems", "model", "models", "process",
    "theory", "method", "methods", "technique", "techniques", "analysis",
    "design", "basics", "basic", "introduction", "fundamentals", "fundamental",
    "engineering", "computer", "science", "rtu", "exam", "paper", "syllabus",
    "hi", "hey", "hello", "search", "please", "pls", "ok", "okay", "condition", "conditions",
}


def _normalize_text(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .lower()
        .strip()
    )


def _tokenize_terms(text: str) -> List[str]:
    norm = _normalize_text(text)
    raw = re.findall(r"[a-z0-9][a-z0-9\+\-#\.]*", norm)
    out: List[str] = []
    for tok in raw:
        tok = tok.strip(".")
        if not tok:
            continue
        if tok.endswith("'s") and len(tok) > 3:
            tok = tok[:-2]
        if len(tok) >= 3 or tok in _SHORT_VALID_TOKENS:
            if tok not in _STOPWORDS:
                out.append(tok)
    return list(dict.fromkeys(out))


class TopicPaperVerifier:
    """Indexes RTU syllabus topics and evaluates query-to-syllabus & paper grounding."""

    def __init__(self, index_dir: Path = settings.INDEXES_DIR):
        self.index_dir = index_dir
        self.cache_npz = index_dir / "topic_embeddings.npz"
        self.cache_meta = index_dir / "topic_meta.json"
        self.topics: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.vocab_tokens: set = set()
        self._is_loaded = False

    def load(self, db: Session, dense_model: Any, force_rebuild: bool = False) -> None:
        """Loads or builds normalized BGE embeddings for all RTU syllabus topics."""
        rows = (
            db.query(Topic, Unit, Subject)
            .join(Unit, Unit.id == Topic.unit_id)
            .join(Subject, Subject.id == Unit.subject_id)
            .order_by(Subject.short_name, Unit.unit_no, Topic.id)
            .all()
        )

        # Map subject_id -> branches & years via offerings
        subj_branches: Dict[str, List[str]] = {}
        subj_years: Dict[str, List[int]] = {}
        for off in db.query(SubjectOffering).all():
            if off.branch and off.branch.code:
                subj_branches.setdefault(off.subject_id, [])
                if off.branch.code not in subj_branches[off.subject_id]:
                    subj_branches[off.subject_id].append(off.branch.code)
            if off.semester:
                yr = (off.semester + 1) // 2
                subj_years.setdefault(off.subject_id, [])
                if yr not in subj_years[off.subject_id]:
                    subj_years[off.subject_id].append(yr)

        records: List[Dict[str, Any]] = []
        doc_texts: List[str] = []
        for topic, unit, subj in rows:
            subj_short = (subj.short_name or subj.code or "RTU").strip()
            subj_name = (subj.name or subj_short).strip()
            unit_title = (unit.title or f"Unit {unit.unit_no}").strip()
            aliases = _normalize_text(topic.aliases or "")
            keyword_phrases = [
                p.strip()
                for p in re.split(r"[,;]+", aliases)
                if len(p.strip()) >= 2
            ]
            # Include key topic name chunks in phrase set
            for p in re.split(r"[,:;()&/]+", _normalize_text(topic.name)):
                p_clean = p.strip()
                if len(p_clean) >= 3 and p_clean not in _STOPWORDS:
                    keyword_phrases.append(p_clean)

            token_set = set(_tokenize_terms(f"{topic.name} {aliases} {unit_title} {subj_name}"))
            doc_text = (
                f"Subject: {subj_name} ({subj_short}, {subj.code}) | "
                f"{unit_title} | Topic: {topic.name} | Keywords: {aliases}"
            )
            records.append({
                "topic_id": topic.id,
                "topic_name": topic.name,
                "aliases": aliases,
                "unit_id": unit.id,
                "unit_no": unit.unit_no,
                "unit_title": unit_title,
                "subject_id": subj.id,
                "subject_short": subj_short,
                "subject_code": subj.code,
                "subject_name": subj_name,
                "branches": subj_branches.get(subj.id, []),
                "years": subj_years.get(subj.id, []),
                "keyword_phrases": list(dict.fromkeys(keyword_phrases)),
                "token_set": sorted(token_set),
                "document_text": doc_text,
            })
            doc_texts.append(doc_text)

        self.topics = records
        vocab: set = {
            "condition", "conditions", "necessary", "insertion", "deletion", "traversal",
            "complexity", "difference", "properties", "deadlock", "algorithm", "algorithms",
        }
        for rec in records:
            for tok in re.findall(r"[a-z]{4,}", f"{rec['topic_name']} {rec['aliases']}".lower()):
                vocab.add(tok)
        self.vocab_tokens = vocab
        if not records or dense_model is None:
            self.embeddings = None
            self._is_loaded = True
            return

        # Reuse cached embeddings if exact topic IDs match
        cached_ok = False
        if not force_rebuild and self.cache_npz.exists() and self.cache_meta.exists():
            try:
                meta = json.loads(self.cache_meta.read_text(encoding="utf-8"))
                if meta.get("topic_ids") == [r["topic_id"] for r in records]:
                    loaded = np.load(str(self.cache_npz))
                    vecs = loaded["embeddings"]
                    if len(vecs) == len(records):
                        self.embeddings = vecs.astype(np.float32)
                        cached_ok = True
            except Exception as exc:
                logger.debug(f"Topic cache reload skipped: {exc}")

        if not cached_ok:
            vecs = dense_model.encode(
                doc_texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            self.embeddings = vecs
            try:
                self.index_dir.mkdir(parents=True, exist_ok=True)
                np.savez(str(self.cache_npz), embeddings=vecs)
                self.cache_meta.write_text(
                    json.dumps({"topic_ids": [r["topic_id"] for r in records]}),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.debug(f"Topic embedding cache save skipped: {exc}")

        self._is_loaded = True
        logger.info(f"[TopicPaperVerifier] Indexed {len(self.topics)} RTU syllabus topics.")

    def _normalize_retrieval_query(self, query: str) -> str:
        """Cleans greetings/fillers, expands compact RTU shorthands, and fixes domain typos."""
        q = query.strip()
        q = re.sub(r"^\s*(?:hi+|hey+|hello+|please|pls)\b[\s,!.:-]*", "", q, flags=re.I)
        q = re.sub(r"^\s*(?:search\s+(?:for|about)\s+|search\s+)", "", q, flags=re.I)
        q = re.sub(r"\bbtrees?\b", "B-Tree", q, flags=re.I)
        q = re.sub(r"\bb\+[\s-]?trees?\b", "B+ Tree", q, flags=re.I)
        q = re.sub(r"\bavltrees?\b", "AVL Tree", q, flags=re.I)

        if not self.vocab_tokens:
            return q or query.strip()

        def _fix_tok(match: re.Match) -> str:
            tok = match.group(0)
            low = tok.lower()
            if len(low) < 6 or low in self.vocab_tokens or low in _STOPWORDS:
                return tok
            same_head = [w for w in self.vocab_tokens if w[0] == low[0] and abs(len(w) - len(low)) <= 3]
            if not same_head:
                return tok
            matches = difflib.get_close_matches(low, same_head, n=3, cutoff=0.77)
            if not matches:
                return tok
            best = max(
                matches,
                key=lambda cand: (
                    difflib.SequenceMatcher(None, low, cand).ratio()
                    - 0.02 * abs(len(cand) - len(low))
                ),
            )
            return best

        corrected = re.sub(r"\b[a-zA-Z]{6,}\b", _fix_tok, q)
        return corrected.strip() or q or query.strip()

    def _subject_matches(self, topic_rec: Dict[str, Any], target_subject: str) -> bool:
        clean = target_subject.replace("SUBJ_", "").strip().upper()
        return (
            topic_rec["subject_short"].upper() == clean
            or topic_rec["subject_id"].upper() == f"SUBJ_{clean}"
            or topic_rec["subject_id"].upper() == target_subject.strip().upper()
            or topic_rec["subject_code"].upper() == target_subject.strip().upper()
        )

    def score_topics(
        self,
        query: str,
        dense_model: Any,
        reranker: Any,
        target_subject: Optional[str] = None,
        branch: Optional[str] = None,
        year: Optional[int] = None,
        max_topics: int = 12,
    ) -> Dict[str, Any]:
        """Scores syllabus topics against the query via dense BGE cosine, lexical keywords, and CrossEncoder."""
        if not self._is_loaded or not self.topics or dense_model is None or self.embeddings is None:
            return {
                "candidates": [],
                "best_cosine": 0.0,
                "best_rerank": -999.0,
                "best_keyword_hits": 0,
                "subject_topic_count": 0,
            }

        q_lower = _normalize_text(query)
        q_tokens = set(_tokenize_terms(q_lower))

        q_vec = dense_model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)[0]
        cosines = np.dot(self.embeddings, q_vec)

        candidate_indices = list(range(len(self.topics)))
        subject_topic_count = 0
        if target_subject:
            subj_indices = [i for i, rec in enumerate(self.topics) if self._subject_matches(rec, target_subject)]
            subject_topic_count = len(subj_indices)
            if subj_indices:
                candidate_indices = subj_indices

        scored: List[Dict[str, Any]] = []
        for i in candidate_indices:
            rec = self.topics[i]
            cos = float(cosines[i])

            # Phrase / alias exact word-boundary hits
            phrase_hits = 0
            matched_phrases: List[str] = []
            for phrase in rec["keyword_phrases"]:
                if len(phrase) < 3 and phrase not in _SHORT_VALID_TOKENS:
                    continue
                if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", q_lower):
                    phrase_hits += 1
                    matched_phrases.append(phrase)

            token_hits = len(q_tokens & set(rec["token_set"])) if q_tokens else 0
            kw_score = phrase_hits * 0.14 + min(token_hits, 4) * 0.05

            context_bonus = 0.0
            if not target_subject:
                if branch and branch.upper() in rec.get("branches", []):
                    context_bonus += 0.02
                if year and year in rec.get("years", []):
                    context_bonus += 0.02

            combined = cos + kw_score + context_bonus
            scored.append({
                "topic_id": rec["topic_id"],
                "topic_name": rec["topic_name"],
                "aliases": rec["aliases"],
                "unit_id": rec["unit_id"],
                "unit_no": rec["unit_no"],
                "unit_title": rec["unit_title"],
                "subject_id": rec["subject_id"],
                "subject_short": rec["subject_short"],
                "subject_code": rec["subject_code"],
                "subject_name": rec["subject_name"],
                "document_text": rec["document_text"],
                "cosine_score": round(cos, 4),
                "phrase_hits": phrase_hits,
                "token_hits": token_hits,
                "matched_phrases": matched_phrases[:5],
                "combined_score": round(combined, 4),
            })

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        shortlist = scored[:15]

        # CrossEncoder verification on top 15 topic candidates
        if shortlist and reranker is not None:
            rerank_in = [dict(item) for item in shortlist]
            reranked = reranker.rerank(query, rerank_in, top_k=len(rerank_in))
            rerank_map = {r["topic_id"]: float(r.get("rerank_score", -999.0)) for r in reranked}
            for item in shortlist:
                item["rerank_score"] = round(rerank_map.get(item["topic_id"], -999.0), 4)
        else:
            for item in shortlist:
                item["rerank_score"] = round((item["cosine_score"] - 0.5) * 10.0, 4)

        shortlist.sort(
            key=lambda x: (
                x["phrase_hits"] * 1.8
                + x["token_hits"] * 0.6
                + x["cosine_score"] * 4.0
                + max(-12.0, min(8.0, x["rerank_score"])) * 0.35
            ),
            reverse=True,
        )

        best_cos = max((x["cosine_score"] for x in shortlist), default=0.0)
        best_rerank = max((x["rerank_score"] for x in shortlist), default=-999.0)
        best_kw = max((x["phrase_hits"] + x["token_hits"] for x in shortlist), default=0)

        # Retain ALL closely related syllabus topics matching the query without unrelated cross-subject noise
        related_topics: List[Dict[str, Any]] = []
        if shortlist:
            top_cos = shortlist[0]["cosine_score"]
            top_subj_id = shortlist[0]["subject_id"]
            top_strong = shortlist[0]["rerank_score"] >= -6.0 or shortlist[0]["phrase_hits"] > 0
            for idx, item in enumerate(shortlist[:max_topics]):
                same_subj_sibling = (
                    top_strong
                    and item["subject_id"] == top_subj_id
                    and item["cosine_score"] >= 0.60
                    and (top_cos - item["cosine_score"]) <= 0.16
                )
                is_related = (
                    (item["phrase_hits"] > 0 and item["rerank_score"] >= -9.0)
                    or (item["token_hits"] >= 2 and item["rerank_score"] >= -8.0)
                    or item["rerank_score"] >= -4.5
                    or same_subj_sibling
                    or (item["cosine_score"] >= 0.58 and item["rerank_score"] >= -6.5 and (top_cos - item["cosine_score"]) <= 0.14)
                    or (idx == 0 and item["cosine_score"] >= 0.60 and item["rerank_score"] >= -6.5)
                )
                if is_related:
                    related_topics.append(item)

        return {
            "candidates": shortlist,
            "related_topics": related_topics,
            "best_cosine": best_cos,
            "best_rerank": best_rerank,
            "best_keyword_hits": best_kw,
            "subject_topic_count": subject_topic_count,
        }

    def fetch_topic_linked_questions(
        self,
        db: Session,
        topic_ids: List[str],
        existing_qids: set,
        target_subject: Optional[str] = None,
        limit: int = 8,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Pulls verified approved past-paper questions directly linked to matched topic_ids."""
        if not topic_ids:
            return [], []

        query = (
            db.query(Question, Paper, Subject, Topic, Unit)
            .join(Paper, Paper.id == Question.paper_id)
            .join(Subject, Subject.id == Paper.subject_id)
            .outerjoin(Topic, Topic.id == Question.topic_id)
            .outerjoin(Unit, Unit.id == Topic.unit_id)
            .filter(
                Question.status == "approved",
                Question.topic_id.in_(topic_ids),
            )
            .order_by(Paper.year.desc())
        )
        if target_subject:
            clean = target_subject.replace("SUBJ_", "").strip().upper()
            query = query.filter(
                (Subject.short_name == clean)
                | (Subject.id == f"SUBJ_{clean}")
                | (Subject.code == target_subject)
            )

        extra_questions: List[Dict[str, Any]] = []
        extra_citations: List[Dict[str, Any]] = []
        for q, paper, subj, topic, unit in query.limit(limit * 3).all():
            if q.id in existing_qids or not is_meaningful_question(q.text):
                continue
            existing_qids.add(q.id)
            sem = subj.offerings[0].semester if subj and subj.offerings else 3
            extra_questions.append({
                "question_id": q.id,
                "paper_id": paper.id,
                "subject_id": subj.id,
                "subject_short": subj.short_name,
                "subject": subj.name,
                "subject_code": subj.code,
                "semester": sem,
                "year": paper.year or 2022,
                "section": q.part or "Part B",
                "unit": f"Unit {unit.unit_no}" if unit else "Unit 1",
                "topic_id": q.topic_id,
                "topic_name": topic.name if topic else "Core Topic",
                "question_number": q.id.split("_")[-1] if "_" in q.id else "1",
                "marks": q.marks or 8,
                "question_text": q.text,
                "source_pdf": paper.source_pdf,
                "page_number": q.page_number or 1,
                "rerank_score": -1.5,
            })
            extra_citations.append({
                "question_id": q.id,
                "paper_name": f"RTU {subj.short_name} ({paper.year or ''})",
                "subject": subj.short_name,
                "subject_short": subj.short_name,
                "subject_id": subj.id,
                "subject_code": subj.code,
                "year": paper.year or 2022,
                "semester": sem,
                "section": q.part or "Part B",
                "part": q.part or "Part B",
                "marks": q.marks or 8,
                "source_pdf": paper.source_pdf,
                "page_number": q.page_number or 1,
                "question_text": q.text,
                "rerank_score": -1.5,
            })
            if len(extra_questions) >= limit:
                break

        return extra_questions, extra_citations

    def verify_and_retrieve(
        self,
        db: Session,
        pipeline: Any,
        query: str,
        target_subject: Optional[str] = None,
        branch: Optional[str] = None,
        year: Optional[int] = None,
        filter_kwargs: Optional[Dict[str, Any]] = None,
        top_k_questions: int = 8,
    ) -> Dict[str, Any]:
        """
        Tests query relevance against RTU syllabus topics AND past-paper questions,
        returning is_rtu_related, related_topics, and expanded question/citation context.
        """
        filter_kwargs = filter_kwargs or {}
        if not self._is_loaded:
            dense = getattr(pipeline, "dense_indexer", None)
            if dense and dense.model is None:
                dense.load_model()
            self.load(db, dense.model if dense else None)

        dense_model = getattr(getattr(pipeline, "dense_indexer", None), "model", None)
        reranker = getattr(pipeline, "reranker", None)
        effective_query = self._normalize_retrieval_query(query)

        # 1. Search & score all syllabus topics (for target_subject if specified, else across RTU)
        topic_eval = self.score_topics(
            query=effective_query,
            dense_model=dense_model,
            reranker=reranker,
            target_subject=target_subject,
            branch=branch,
            year=year,
            max_topics=12,
        )

        # Also check global topics when target_subject has 0 topics or scores poorly (cross-subject check)
        global_topic_eval = topic_eval
        if target_subject and (topic_eval["subject_topic_count"] == 0 or topic_eval["best_rerank"] < -6.0):
            global_topic_eval = self.score_topics(
                query=effective_query,
                dense_model=dense_model,
                reranker=reranker,
                target_subject=None,
                branch=branch,
                year=year,
                max_topics=12,
            )

        # 2. Search past-paper question archive via Hybrid RetrievalPipeline
        q_res = pipeline.retrieve(
            query=effective_query,
            subject=target_subject,
            semester=filter_kwargs.get("semester"),
            year_from=filter_kwargs.get("year_from"),
            year_to=filter_kwargs.get("year_to"),
            section=filter_kwargs.get("section"),
            marks=filter_kwargs.get("marks"),
            top_k=max(top_k_questions, 8),
            rerank_candidates_k=24,
        )
        raw_questions = q_res.get("results", [])
        raw_citations = q_res.get("citations", [])

        # Subject isolation on questions when target_subject active
        if target_subject:
            t_clean = target_subject.replace("SUBJ_", "").strip().upper()
            q_matches = [
                rq for rq in raw_questions
                if (
                    rq.get("subject_short", "").upper() == t_clean
                    or rq.get("subject", "").upper() == t_clean
                    or rq.get("subject_id", "").upper() == f"SUBJ_{t_clean}"
                )
            ]
            c_matches = [
                c for c in raw_citations
                if (
                    c.get("subject_short", "").upper() == t_clean
                    or c.get("subject", "").upper() == t_clean
                    or c.get("subject_id", "").upper() == f"SUBJ_{t_clean}"
                )
            ]
        else:
            q_matches = list(raw_questions)
            c_matches = list(raw_citations)

        best_q_score = max((rq.get("rerank_score", -999.0) for rq in q_matches), default=-999.0)
        q_is_fallback = any(rq.get("rerank_fallback", False) for rq in q_matches)

        # 3. Expand question set with topic-linked DB questions ONLY for strong matched topics
        related_topics = topic_eval["related_topics"]
        if not related_topics and topic_eval["subject_topic_count"] == 0 and global_topic_eval["related_topics"]:
            related_topics = global_topic_eval["related_topics"]

        strong_topic_ids = [
            t["topic_id"]
            for t in related_topics
            if t["phrase_hits"] > 0 or t["token_hits"] >= 2 or t["rerank_score"] >= -4.5
        ]
        if not strong_topic_ids and related_topics:
            strong_topic_ids = [t["topic_id"] for t in related_topics[:2]]

        matched_topic_ids = list(strong_topic_ids)
        # Also include topics referenced directly by confident question hits
        q_thresh = getattr(settings, "OUT_OF_SCOPE_THRESHOLD", -9.5)
        cit_thresh = getattr(settings, "CITATION_RELEVANCE_THRESHOLD", -6.0)
        for rq in q_matches:
            if not rq.get("rerank_fallback") and rq.get("rerank_score", -999.0) >= cit_thresh and rq.get("topic_id"):
                tid = rq["topic_id"]
                if tid not in matched_topic_ids:
                    matched_topic_ids.append(tid)

        existing_qids = {rq.get("question_id") for rq in q_matches if rq.get("question_id")}
        extra_qs, extra_cits = self.fetch_topic_linked_questions(
            db=db,
            topic_ids=matched_topic_ids,
            existing_qids=existing_qids,
            target_subject=target_subject,
            limit=8,
        )
        # Re-score extra topic-linked questions with CrossEncoder when available so matching years/topics lead
        if extra_qs and reranker is not None:
            for eq in extra_qs:
                eq["document_text"] = (
                    f"{eq.get('subject', '')} ({eq.get('subject_short', '')}) | "
                    f"{eq.get('topic_name', '')} | {eq.get('question_text', '')}"
                )
            scored_extra = reranker.rerank(effective_query, extra_qs, top_k=len(extra_qs))
            extra_map = {e["question_id"]: e for e in scored_extra}
            kept_extra_qs: List[Dict[str, Any]] = []
            kept_extra_cits: List[Dict[str, Any]] = []
            for eq, ec in zip(extra_qs, extra_cits):
                rescored = extra_map.get(eq["question_id"], eq)
                r_sc = rescored.get("rerank_score", -999.0)
                if eq.get("topic_id") in strong_topic_ids or r_sc >= cit_thresh:
                    eq["rerank_score"] = r_sc
                    ec["rerank_score"] = r_sc
                    kept_extra_qs.append(eq)
                    kept_extra_cits.append(ec)
            kept_extra_qs.sort(key=lambda x: x.get("rerank_score", -999.0), reverse=True)
            kept_extra_cits.sort(key=lambda x: x.get("rerank_score", -999.0), reverse=True)
            extra_qs, extra_cits = kept_extra_qs, kept_extra_cits

        # Keep highest-scoring retrieval & topic-linked questions in relevance order
        all_questions = sorted(
            q_matches + extra_qs,
            key=lambda x: (x.get("rerank_score", -999.0) if not x.get("rerank_fallback") else -8.0),
            reverse=True,
        )
        confident_cits = [
            c for c, rq in zip(c_matches, q_matches)
            if not rq.get("rerank_fallback") and rq.get("rerank_score", -999.0) >= cit_thresh
        ]
        if extra_cits and (best_q_score >= cit_thresh or topic_eval["best_keyword_hits"] > 0 or strong_topic_ids):
            combined_cits = extra_cits + confident_cits
            combined_cits.sort(key=lambda x: x.get("rerank_score", -999.0), reverse=True)
            all_citations = combined_cits
        else:
            all_citations = confident_cits

        # 4. Determine RTU-related vs out-of-the-box classification (calibrated empirically on BGE + MiniLM)
        t_rerank = topic_eval["best_rerank"]
        t_cos = topic_eval["best_cosine"]
        t_kw = topic_eval["best_keyword_hits"]
        g_rerank = global_topic_eval["best_rerank"]
        g_cos = global_topic_eval["best_cosine"]
        g_kw = global_topic_eval["best_keyword_hits"]

        topic_strong = (
            (t_kw >= 1 and t_cos >= 0.50 and t_rerank >= -7.8)
            or (t_rerank >= -5.2 and t_cos >= 0.52)
            or (t_cos >= 0.70 and t_rerank >= -8.0)
        )
        global_topic_strong = (
            (g_kw >= 1 and g_cos >= 0.52 and g_rerank >= -7.5)
            or (g_rerank >= -4.8 and g_cos >= 0.54)
            or (g_cos >= 0.72 and g_rerank >= -7.5)
        )
        paper_strong = bool(q_matches) and (not q_is_fallback) and (
            best_q_score >= -7.5
            or (
                best_q_score >= q_thresh
                and ((t_rerank >= -6.5 and t_cos >= 0.50) or (g_rerank >= -6.5 and g_cos >= 0.50))
            )
        )

        # If target_subject is explicitly set and has topics, verify query belongs to RTU (or target subject)
        if target_subject:
            if topic_eval["subject_topic_count"] > 0:
                is_rtu_related = bool(topic_strong or paper_strong or (global_topic_strong and not raw_questions))
            else:
                is_rtu_related = bool(paper_strong or global_topic_strong)
        else:
            is_rtu_related = bool(topic_strong or global_topic_strong or paper_strong)

        # Ensure out-of-scope mismatch guard: if neither topics nor papers match cleanly, clear extra links
        if not is_rtu_related:
            related_topics = []
            all_questions = []
            all_citations = []

        return {
            "is_rtu_related": is_rtu_related,
            "related_topics": related_topics,
            "questions": all_questions[:max(top_k_questions, 8)],
            "citations": all_citations[:max(top_k_questions, 8)],
            "topic_eval": {
                "best_topic_cosine": t_cos,
                "best_topic_rerank": t_rerank,
                "best_topic_keywords": t_kw,
                "global_topic_cosine": g_cos,
                "global_topic_rerank": g_rerank,
                "best_question_rerank": best_q_score,
                "related_topic_count": len(related_topics),
            },
            "retrieval_stats": {
                **(q_res.get("retrieval_stats") or {}),
                "related_topics_found": len(related_topics),
                "topic_linked_questions_added": len(extra_qs),
                "best_topic_cosine": t_cos,
                "best_topic_rerank": t_rerank,
                "best_question_rerank": best_q_score,
                "is_rtu_related": is_rtu_related,
            },
        }
