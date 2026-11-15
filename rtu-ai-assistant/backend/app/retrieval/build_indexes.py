"""
Index Builder CLI for RTU AI Academic Assistant
Executes document preparation and builds both BGE-Small FAISS index
and BM25 keyword index from the SQLite database.
"""

from pathlib import Path
from ..database.session import SessionLocal
from .pipeline import RetrievalPipeline

def main():
    print("=" * 65)
    print("  PHASE 2: BUILDING HYBRID RETRIEVAL INDEXES (FAISS + BM25)")
    print("=" * 65)

    db = SessionLocal()
    pipeline = RetrievalPipeline()
    pipeline.build_indexes(db)
    db.close()

    print("\n[+] Verification: Testing sample query through Hybrid Pipeline...")
    db = SessionLocal()
    pipeline.load_indexes(db)
    
    test_query = "Explain bubble sort and quick sort algorithm with complexity"
    res = pipeline.retrieve(test_query, subject="DSA", top_k=3)
    
    print(f"\nQuery: {test_query}")
    print(f"Retrieved {res['results_count']} top candidates:")
    for i, r in enumerate(res["results"], 1):
        print(f"  [{i}] CrossEncoder Score: {r['rerank_score']:.3f} | {r['subject_short']} ({r['year']}) {r['section']}")
        print(f"      Question: {r['question_text'][:80]}...")
        print(f"      Source: {r['source_pdf']} (Page {r['page_number']})")
    
    db.close()
    print("\n" + "=" * 65)
    print("  PHASE 2 INDEXING & VERIFICATION COMPLETE")
    print("=" * 65)

if __name__ == "__main__":
    main()

