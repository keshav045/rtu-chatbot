"""
BM25 Keyword Indexer for Technical Queries, Subject Codes and Question Numbers
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

def technical_tokenize(text: str) -> List[str]:
    """
    Tokenizes text while preserving subject codes (3E1202), question numbers (Q3(a)),
    acronyms (AVL, BST, DBMS), marks numbers (2, 8, 15), and technical keywords.
    """
    clean_text = text.lower()
    # Match alphanumeric sequences, words with hyphens/parentheses (e.g. 3e1202, q3(a), b-tree, part-c)
    tokens = re.findall(r"[a-z0-9]+(?:[\(\)\-_][a-z0-9]+)*", clean_text)
    return tokens

class BM25Indexer:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_path = index_dir / "bm25_index.pkl"
        self.bm25 = None
        self.id_mapping = []

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.id_mapping = [doc["question_id"] for doc in documents]

        corpus = [technical_tokenize(doc["document_text"]) for doc in documents]
        self.bm25 = BM25Okapi(corpus)

        data = {
            "bm25": self.bm25,
            "id_mapping": self.id_mapping
        }
        with open(self.index_path, "wb") as f:
            pickle.dump(data, f)

        print(f"BM25 index built with {len(self.id_mapping)} documents at {self.index_path}")
        return len(self.id_mapping)

    def load_index(self):
        if not self.index_path.exists():
            raise FileNotFoundError(f"BM25 index not found at {self.index_path}")

        with open(self.index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.id_mapping = data["id_mapping"]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if self.bm25 is None or not self.id_mapping:
            self.load_index()

        tokens = technical_tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            if score > 0.0:  # Only include non-zero keyword matches
                results.append((self.id_mapping[idx], score))

        return results

