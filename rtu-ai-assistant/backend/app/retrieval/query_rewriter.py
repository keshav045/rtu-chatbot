"""
Conversational Memory & Contextual Query Rewriter
Translates multi-turn follow-up queries into standalone retrieval queries
while preserving student conversation context and tracking active topics.
"""

import re
from typing import List, Dict, Any, Optional

try:
    from app.services.intent_service import SUBJECT_ALIASES
except ImportError:
    SUBJECT_ALIASES = [
        ("data structures", "DSA"), ("dsa", "DSA"),
        ("database", "DBMS"), ("dbms", "DBMS"),
        ("networks", "DCCN"), ("dccn", "DCCN"),
        ("automata", "TOC"), ("toc", "TOC"),
        ("mathematics", "AEM"), ("aem", "AEM"),
        ("digital electronics", "DE"), ("de", "DE")
    ]

FOLLOW_UP_INDICATORS = [
    r"\bit\b", r"\bthis\b", r"\bthat\b", r"\bthem\b",
    r"\bthe same\b", r"\bprevious\b", r"\babove\b",
    r"\bquestion\s*\d+", r"\bpart\s*[abc]\b",
    r"\bfor\s*\d+\s*marks\b", r"\bexplain\s*more\b",
    r"\bgive\s*(an)?\s*example\b", r"\bwhat\s*about\b",
    r"\bcode\s*for\s*it\b", r"\bnow\s*explain\b",
    r"\bhow\s*to\s*solve\s*it\b", r"\balgorithm\s*for\s*this\b"
]

class ContextualQueryRewriter:
    def __init__(self):
        self.follow_up_regex = re.compile("|".join(FOLLOW_UP_INDICATORS), re.I)

    def is_follow_up(self, query: str) -> bool:
        """Checks if a student message is a follow-up query."""
        q = query.strip()
        # Explicit follow-up indicators (e.g., 'it', 'this', 'explain more', 'give example')
        if self.follow_up_regex.search(q):
            return True
        # Very short continuation phrases like 'why?', 'how so?', 'and this?'
        words = q.lower().split()
        if len(words) <= 3 and any(w in words for w in ["why", "how", "what", "more", "next", "and"]):
            return True
        return False

    def extract_active_context(self, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Scans recent messages to extract the active subject, year, topic, and last mentioned question.
        """
        context = {
            "subject": None,
            "year": None,
            "topic": None,
            "question_number": None,
            "marks": None
        }

        # Check in reverse from latest assistant/user message
        for msg in reversed(conversation_history):
            content = msg.get("content", "")

            # Detect subject
            if not context["subject"]:
                for phrase, code in SUBJECT_ALIASES:
                    if re.search(rf"\b{re.escape(phrase)}\b", content, re.I):
                        context["subject"] = code
                        break

            # Detect year
            if not context["year"]:
                year_match = re.search(r"\b(20[12]\d)\b", content)
                if year_match:
                    context["year"] = int(year_match.group(1))

            # Detect marks
            if not context["marks"]:
                marks_match = re.search(r"\b(\d+)\s*marks?\b", content, re.I)
                if marks_match:
                    context["marks"] = int(marks_match.group(1))

            # Detect question number
            if not context["question_number"]:
                q_match = re.search(r"\b(Q\.?\s*\d+|question\s*\d+)\b", content, re.I)
                if q_match:
                    context["question_number"] = q_match.group(1)

            # Detect primary technical topic (e.g. AVL tree, QuickSort, Normalization, etc.)
            if not context["topic"]:
                for term in [
                    "quick sort", "bubble sort", "merge sort", "heap sort", "avl tree",
                    "binary search tree", "linked list", "stack", "queue", "dijkstra",
                    "normalization", "bcnf", "3nf", "2pl", "two-phase locking", "acid",
                    "pumping lemma", "dfa", "nfa", "turing machine", "cfg", "pda",
                    "master-slave", "flip flop", "k-map", "boolean algebra",
                    "runge-kutta", "newton-raphson", "simpson", "laplace"
                ]:
                    if term in content.lower():
                        context["topic"] = term.title()
                        break

        return context

    def rewrite_query(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        """
        Transforms a follow-up query into a standalone retrieval query using conversation context.
        """
        if not conversation_history or not self.is_follow_up(user_query):
            return user_query

        ctx = self.extract_active_context(conversation_history)

        # Check if user explicitly specified a new subject or year in the current query
        for phrase, code in SUBJECT_ALIASES:
            if re.search(rf"\b{re.escape(phrase)}\b", user_query, re.I):
                ctx["subject"] = code
                break

        components = []
        if ctx["subject"]:
            components.append(f"Subject: {ctx['subject']}")
        if ctx["year"]:
            components.append(f"Year: {ctx['year']}")
        if ctx["topic"]:
            components.append(f"Topic: {ctx['topic']}")
        if ctx["question_number"]:
            components.append(f"{ctx['question_number']}")

        if not components:
            return user_query

        context_prefix = " ".join(components)
        rewritten = f"{context_prefix} - {user_query}"
        return rewritten

    # Convenient alias
    def rewrite(self, user_query: str, conversation_history: List[Dict[str, str]]) -> str:
        return self.rewrite_query(user_query, conversation_history)

# Convenient alias for class
QueryRewriter = ContextualQueryRewriter

