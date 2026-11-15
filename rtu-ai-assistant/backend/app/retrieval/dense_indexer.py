"""
Dense Vector Indexer using BAAI/bge-small-en-v1.5 and FAISS
Cosine similarity retrieval via normalized embeddings and IndexFlatIP.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss

logger = logging.getLogger("rtu.retrieval")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

class DenseIndexer:
    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_path = index_dir / "rtu_faiss_bge.index"
        self.mapping_path = index_dir / "faiss_id_mapping.json"
        self.model = None
        self.index = None
        self.id_mapping = []  # list of question_id corresponding to vector index i

    def load_model(self):
        """Loads SentenceTransformer eagerly into memory."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(MODEL_NAME)
                logger.info(f"SentenceTransformer ({MODEL_NAME}) loaded into memory successfully.")
            except Exception as e:
                logger.error(f"Error loading SentenceTransformer: {e}")
                raise e

    def build_index(self, documents: List[Dict[str, Any]]) -> int:
        self.load_model()
        self.index_dir.mkdir(parents=True, exist_ok=True)

        texts = [doc["document_text"] for doc in documents]
        self.id_mapping = [doc["question_id"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        faiss.write_index(self.index, str(self.index_path))
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.id_mapping, f, indent=2)

        print(f"Dense FAISS index built with {self.index.ntotal} vectors at {self.index_path}")
        return self.index.ntotal

    def load_index(self):
        """Eagerly loads FAISS vector index, mappings, and SentenceTransformer embedding model."""
        if not self.index_path.exists() or not self.mapping_path.exists():
            raise FileNotFoundError(f"FAISS index or mapping not found in {self.index_dir}")

        self.index = faiss.read_index(str(self.index_path))
        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.id_mapping = json.load(f)

        # Eagerly load the neural model into memory at startup
        self.load_model()
        logger.info(f"Loaded FAISS dense index ({self.index.ntotal} vectors) and model {MODEL_NAME} eagerly into RAM.")

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        if self.index is None or not self.id_mapping:
            self.load_index()

        if self.model is None:
            self.load_model()

        query_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1 and idx < len(self.id_mapping):
                results.append((self.id_mapping[idx], float(score)))

        return results
