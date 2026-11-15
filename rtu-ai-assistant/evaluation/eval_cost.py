"""
Cost and Token Economics Evaluation Module (eval_cost.py).
Evaluates token consumption and financial costs across LLM providers:
- Groq (Llama-3.3-70B)
- Gemini 1.5 Flash
- Local / In-Process Offline Mode
Projects operational budget for 1,000 to 10,000 RTU engineering students.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from harness import EvaluationHarness
from metric_registry import estimate_cost

COST_BENCHMARK_PROMPTS = [
    {"subject": "DSA", "query": "Explain QuickSort partition algorithm with step by step trace and average/worst case recurrence"},
    {"subject": "OS", "query": "Explain Banker algorithm for deadlock avoidance with safety algorithm example matrix"},
    {"subject": "DBMS", "query": "Explain 1NF, 2NF, 3NF, and BCNF normalization with functional dependencies"},
    {"subject": "CN", "query": "Explain Go-Back-N and Selective Repeat ARQ protocols with window size formulas"},
    {"subject": "TOC", "query": "Construct minimal DFA for binary strings divisible by 3 and write transition table"}
]

def run_cost_eval() -> Dict[str, Any]:
    harness = EvaluationHarness()
    harness.setup()

    print("\n[eval_cost] Estimating token usage and API costs across providers...")

    total_prompt_chars = 0
    total_completion_chars = 0

    for item in COST_BENCHMARK_PROMPTS:
        # Retrieve context
        ret = harness.pipeline.retrieve(query=item["query"], subject=item["subject"], top_k=5)
        ctx = ret.get("context_string", "")
        prompt_text = f"Context: {ctx}\nQuestion: {item['query']}"
        total_prompt_chars += len(prompt_text)

        # Generate sample reply
        res = harness.chat_service.chat(
            db=harness.db,
            user_message=item["query"],
            detected_subject=item["subject"],
            provider_name="offline"
        )
        reply = res.get("assistant_reply", "")
        total_completion_chars += len(reply)

    # Convert characters to approximate tokens (1 token ~ 4 chars)
    prompt_tokens = int(total_prompt_chars / 4.0)
    completion_tokens = int(total_completion_chars / 4.0)
    num_queries = len(COST_BENCHMARK_PROMPTS)

    avg_prompt_tokens = int(prompt_tokens / num_queries)
    avg_completion_tokens = int(completion_tokens / num_queries)

    # Cost calculation across providers
    groq_cost = estimate_cost(prompt_tokens, completion_tokens, provider="groq")
    gemini_cost = estimate_cost(prompt_tokens, completion_tokens, provider="gemini")
    offline_cost = estimate_cost(prompt_tokens, completion_tokens, provider="offline")

    # Scaling projections for 1,000 queries and 10,000 queries
    groq_1k = round(groq_cost["cost_usd"] * (1000.0 / num_queries), 3)
    gemini_1k = round(gemini_cost["cost_usd"] * (1000.0 / num_queries), 3)

    summary = {
        "benchmark": "eval_cost",
        "sample_queries": num_queries,
        "avg_prompt_tokens_per_query": avg_prompt_tokens,
        "avg_completion_tokens_per_query": avg_completion_tokens,
        "avg_total_tokens_per_query": avg_prompt_tokens + avg_completion_tokens,
        "groq_cost_per_query_usd": round(groq_cost["cost_usd"] / num_queries, 6),
        "gemini_cost_per_query_usd": round(gemini_cost["cost_usd"] / num_queries, 6),
        "offline_cost_usd": 0.0,
        "projected_cost_1000_queries": {
            "groq_usd": groq_1k,
            "gemini_usd": gemini_1k,
            "offline_usd": 0.0
        },
        "projected_cost_10000_queries": {
            "groq_usd": round(groq_1k * 10, 2),
            "gemini_usd": round(gemini_1k * 10, 2),
            "offline_usd": 0.0
        }
    }

    print("\n=================== COST & TOKEN ECONOMICS ===================")
    print(f"Average Prompt Tokens / Query:     {summary['avg_prompt_tokens_per_query']} tokens")
    print(f"Average Completion Tokens / Query: {summary['avg_completion_tokens_per_query']} tokens")
    print(f"Average Total Tokens / Query:      {summary['avg_total_tokens_per_query']} tokens")
    print("--------------------------------------------------------------")
    print(f"Groq (Llama-3.3-70B) Cost / Query: ${summary['groq_cost_per_query_usd']:.6f} USD")
    print(f"Gemini 1.5 Flash Cost / Query:     ${summary['gemini_cost_per_query_usd']:.6f} USD")
    print(f"Local / In-Process Cost:           $0.000000 USD (FREE)")
    print("--------------------------------------------------------------")
    print(f"Projected Cost for 1,000 Queries:")
    print(f"  - Groq:   ${summary['projected_cost_1000_queries']['groq_usd']} USD")
    print(f"  - Gemini: ${summary['projected_cost_1000_queries']['gemini_usd']} USD")
    print(f"  - Offline: $0.00 USD")
    print(f"Projected Cost for 10,000 Queries:")
    print(f"  - Groq:   ${summary['projected_cost_10000_queries']['groq_usd']} USD")
    print(f"  - Gemini: ${summary['projected_cost_10000_queries']['gemini_usd']} USD")
    print("==============================================================\n")

    return summary

if __name__ == "__main__":
    run_cost_eval()
