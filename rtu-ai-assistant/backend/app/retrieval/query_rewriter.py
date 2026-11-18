"""
Conversational Memory & Contextual Query Rewriter
Translates multi-turn follow-up queries into standalone retrieval queries
while preserving student conversation context, Hinglish normalization,
RTU synonym expansion, and Part A/B/C hint detection.
"""

import re
from typing import List, Dict, Any, Optional

# Hinglish phrase mappings (Hinglish -> English)
HINGLISH_MAP = [
    (r"\bkya hai\b", "what is"),
    (r"\bkya hota hai\b", "what is"),
    (r"\bkaise kaam karta hai\b", "how does it work"),
    (r"\bkaise karte hain\b", "how to do"),
    (r"\bkaise hota hai\b", "how does it happen"),
    (r"\bsamjhao\b", "explain"),
    (r"\bsamjhaye\b", "explain"),
    (r"\bexplain karo\b", "explain"),
    (r"\bexplain karte\b", "explain"),
    (r"\bbatao\b", "explain"),
    (r"\bbtao\b", "explain"),
    (r"\blikho\b", "write"),
    (r"\blikhiye\b", "write"),
    (r"\bparibhasha\b", "definition"),
    (r"\bparibhasha dijiye\b", "define"),
    (r"\bdefinition do\b", "definition of"),
    (r"\bacchabhai\b", ""),
    (r"\bbhai\b", ""),
    (r"\byaar\b", ""),
    (r"\bkripya\b", "please"),
    (r"\bmujhe batao\b", "tell me"),
    (r"\bplease batao\b", "please explain"),
    (r"\btheory\b", "theory"),
    (r"\bdetail mein\b", "in detail"),
    (r"\bvistaar se\b", "in detail"),
    (r"\bsaral shabdon mein\b", "in simple terms"),
    (r"\bhi\b", ""),
]

# RTU Subject synonym expansion
RTU_SYNONYMS = {
    "dsa": "data structures algorithms",
    "data structure": "data structures algorithms DSA",
    "ds": "data structures DSA",
    "dbms": "database management system DBMS",
    "database": "database management system DBMS",
    "dccn": "data communication computer networks DCCN",
    "computer networks": "data communication computer networks DCCN",
    "cn": "computer networks DCCN",
    "toc": "theory of computation automata TOC",
    "automata": "theory of computation automata TOC",
    "flat": "formal languages automata theory TOC",
    "de": "digital electronics DE",
    "digital electronics": "digital electronics DE",
    "aem": "advanced engineering mathematics AEM",
    "engineering mathematics": "advanced engineering mathematics AEM",
    "maths": "mathematics AEM",
    "os": "operating system OS",
    "operating system": "operating system OS",
    "se": "software engineering SE",
    "software engineering": "software engineering SE",
    "coa": "computer organization architecture COA",
    "computer organization": "computer organization architecture COA",
    "oops": "object oriented programming OOP",
    "oop": "object oriented programming OOP",
    "java": "java programming OOP",
    "python": "python programming",
    "c programming": "c language programming",
    "microprocessor": "microprocessor microcontroller",
}

# Part marker patterns
PART_A_RE = re.compile(r"\bpart\s*[-:]?\s*a\b|\bshort\s*questions?\b|\b2\s*marks?\b", re.I)
PART_B_RE = re.compile(r"\bpart\s*[-:]?\s*b\b|\b6\s*marks?\b|\bmedium\s*answer\b", re.I)
PART_C_RE = re.compile(r"\bpart\s*[-:]?\s*c\b|\b10\s*marks?\b|\blong\s*answer\b|\bdescribe\s+in\s+detail\b", re.I)

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
        self._hinglish_compiled = [
            (re.compile(pat, re.I), repl) for pat, repl in HINGLISH_MAP
        ]

    def normalize_hinglish(self, query: str) -> str:
        """Convert common Hinglish phrases to English equivalents."""
        q = query
        for pattern, replacement in self._hinglish_compiled:
            q = pattern.sub(replacement, q)
        # Normalize multiple spaces
        q = re.sub(r"\s{2,}", " ", q).strip()
        return q

    def expand_synonyms(self, query: str) -> str:
        """Expand RTU subject abbreviations/synonyms for better retrieval."""
        q_lower = query.lower()
        expansions = []
        for term, expansion in RTU_SYNONYMS.items():
            # Only expand if term appears as a standalone word
            if re.search(rf"\b{re.escape(term)}\b", q_lower):
                expansions.append(expansion)
        if expansions:
            return query + " " + " ".join(expansions)
        return query

    def detect_part_hint(self, query: str) -> Optional[str]:
        """Return 'A', 'B', or 'C' if query contains a part hint, else None."""
        if PART_C_RE.search(query):
            return "C"
        if PART_B_RE.search(query):
            return "B"
        if PART_A_RE.search(query):
            return "A"
        return None

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
        Also applies Hinglish normalization before rewriting.
        """
        # Step 1: Normalize Hinglish
        normalized = self.normalize_hinglish(user_query)

        if not conversation_history or not self.is_follow_up(normalized):
            return normalized

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

    def rewrite_full(self, user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Returns a dict with:
          - query: the fully rewritten + synonym-expanded query for retrieval
          - hint_part: detected Part A/B/C or None
          - original: the original user_query unchanged
        """
        normalized = self.normalize_hinglish(user_query)
        rewritten = self.rewrite_query(user_query, conversation_history)
        expanded = self.expand_synonyms(rewritten)
        hint_part = self.detect_part_hint(user_query)
        return {
            "query": expanded,
            "hint_part": hint_part,
            "original": user_query,
        }

# Convenient alias for class
QueryRewriter = ContextualQueryRewriter

