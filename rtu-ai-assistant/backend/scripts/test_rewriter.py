import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))
from app.retrieval.query_rewriter import QueryRewriter

qr = QueryRewriter()
tests = [
    ('normalization explain karo detail mein', []),
    ('OS scheduling kya hota hai bhai', []),
    ('What is DBMS normalization Part B', []),
    ('AVL tree rotations 10 marks', []),
    ('explain it with example', [{'role':'user','content':'Tell me about quicksort DSA'}]),
    ('kaise kaam karta hai', [{'role':'user','content':'DCCN TCP/IP protocols'}]),
]
for q, h in tests:
    r = qr.rewrite_full(q, h)
    q_out = r["query"][:110]
    part = r["hint_part"]
    print(f"INPUT:      {q}")
    print(f"  query:    {q_out}")
    print(f"  part:     {part}")
    print()
