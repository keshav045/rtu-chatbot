"""
Query Intent Classifier & Router for RTU AI Academic Assistant
Understands what the student wants and routes to the appropriate service/action:
- GREETING: Conversational hello/greeting
- CLOSING: Thanks, acknowledgment, or saying bye
- HELP_CAPABILITIES: Asking how the assistant works, what it can do, features, or asking for guidance
- STUDY_PLAN: Time-constrained revision planning (e.g., 6 hours left, crash course)
- TOPIC_IMPORTANCE: Repeated questions, topic frequency, weightage
- PAST_YEAR_QUESTIONS: Past exam papers / PYQ lookup
- SUBJECT_SELECT: Stating or selecting an RTU subject (e.g., "DSA", "DBMS", "computer networks")
- OUT_OF_SYLLABUS: Non-academic or off-topic questions
- EXAM_QUESTION: Technical RTU engineering concept or question (Full RAG Pipeline)
"""

import re
from enum import Enum
from typing import Dict, Any, Optional

class IntentType(str, Enum):
    SECURITY_VIOLATION = "SECURITY_VIOLATION"
    GREETING = "GREETING"
    CLOSING = "CLOSING"
    HELP_CAPABILITIES = "HELP_CAPABILITIES"
    STUDY_PLAN = "STUDY_PLAN"
    TOPIC_IMPORTANCE = "TOPIC_IMPORTANCE"
    PAST_YEAR_QUESTIONS = "PAST_YEAR_QUESTIONS"
    QUIZ_REQUEST = "QUIZ_REQUEST"
    SUBJECT_SELECT = "SUBJECT_SELECT"
    OUT_OF_SYLLABUS = "OUT_OF_SYLLABUS"
    EXAM_QUESTION = "EXAM_QUESTION"

# Common technical terms in RTU B.Tech engineering syllabus
TECH_TERMS = [
    "tree", "sort", "algorithm", "graph", "database", "sql", "normalization",
    "matrix", "automata", "flip flop", "cpu", "memory", "paging", "cache",
    "hashing", "hash", "dijkstra", "stack", "queue", "linked list", "recursion",
    "array", "pointer", "b-tree", "avl", "turing", "dfa", "nfa", "k-map",
    "binary search", "bfs", "dfs", "spanning tree", "kruskal", "prim",
    "quicksort", "mergesort", "heapsort", "complexity", "big-o", "np-complete",
    "relational algebra", "er model", "bcnf", "3nf", "deadlock", "process",
    "thread", "semaphore", "virtual memory", "scheduling", "tcp", "udp",
    "osi", "ip", "routing", "crc", "hamming", "parity", "compiler",
    "parsing", "lexical", "syntax", "grammar", "ll(1)", "lr", "laplace",
    "fourier", "runge-kutta", "simpson", "newton-raphson"
]

# Canonical RTU Subject Mapping with common names and aliases
# Ordered by specificity (longer phrases first)
SUBJECT_ALIASES = [
    ("data structures and algorithms", "DSA"),
    ("data structures & algorithms", "DSA"),
    ("data structures", "DSA"),
    ("data structure", "DSA"),
    ("database management systems", "DBMS"),
    ("database management system", "DBMS"),
    ("database management", "DBMS"),
    ("database systems", "DBMS"),
    ("data communication and computer networks", "DCCN"),
    ("data communication & computer networks", "DCCN"),
    ("computer networks", "DCCN"),
    ("computer network", "DCCN"),
    ("theory of computation", "TOC"),
    ("automata theory", "TOC"),
    ("automata", "TOC"),
    ("advanced engineering mathematics", "AEM"),
    ("engineering mathematics", "AEM"),
    ("engineering maths", "AEM"),
    ("digital electronics", "DE"),
    ("object oriented programming", "OOP"),
    ("object oriented", "OOP"),
    ("software engineering", "SE"),
    ("discrete mathematics structure", "DMS"),
    ("discrete mathematics", "DMS"),
    ("discrete math", "DMS"),
    ("microprocessor and interfaces", "MI"),
    ("microprocessor", "MI"),
    ("microprocessors", "MI"),
    ("operating systems", "OS"),
    ("operating system", "OS"),
    ("machine learning", "ML"),
    ("artificial intelligence", "AI"),
    ("compiler design", "CD"),
    ("cloud computing", "CC"),
    ("information system security", "ISS"),
    ("internet of things", "IOT"),
    ("deep learning", "DL"),
    ("distributed systems", "DS"),
    ("digital image processing", "DIP"),
    ("blockchain technologies", "BC"),
    ("blockchain", "BC"),
    ("principles of communication", "POC"),
    ("technical communication", "TC"),
    ("managerial economics", "MEFA"),
    ("basic electrical engineering", "BEE"),
    ("basic civil engineering", "BCE"),
    ("basic mechanical engineering", "BME"),
    ("engineering physics", "PHY"),
    ("engineering chemistry", "CHEM"),
    ("physics", "PHY"),
    ("chemistry", "CHEM"),
    # Exact acronyms / short codes
    ("dsa", "DSA"),
    ("dbms", "DBMS"),
    ("dccn", "DCCN"),
    ("toc", "TOC"),
    ("aem", "AEM"),
    ("de", "DE"),
    ("oops", "OOP"),
    ("oop", "OOP"),
    ("se", "SE"),
    ("dms", "DMS"),
    ("mi", "MI"),
    ("mp", "MI"),
    ("os", "OS"),
    ("ml", "ML"),
    ("ai", "AI"),
    ("cd", "CD"),
    ("cc", "CC"),
    ("iss", "ISS"),
    ("iot", "IOT"),
    ("dl", "DL"),
    ("ds", "DS"),
    ("dip", "DIP"),
    ("bc", "BC"),
    ("poc", "POC"),
    ("tc", "TC"),
    ("mefa", "MEFA"),
    ("bee", "BEE"),
    ("bce", "BCE"),
    ("bme", "BME")
]

class IntentRouter:
    @staticmethod
    def extract_entities(query: str) -> Dict[str, Any]:
        """
        Extracts key entities like hours, subject code, and unit number from the query.
        """
        entities: Dict[str, Any] = {
            "hours": None,
            "subject": None,
            "unit": None
        }

        # Extract hours (e.g., "6 hr", "4.5 hours", "2 ghante")
        hours_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:hr|hour|ghante|ghanta)s?', query, re.IGNORECASE)
        if hours_match:
            try:
                entities["hours"] = float(hours_match.group(1))
            except ValueError:
                pass

        # Extract subject using canonical aliases
        q_lower = query.strip().lower()
        for phrase, code in SUBJECT_ALIASES:
            if re.search(rf'\b{re.escape(phrase)}\b', q_lower):
                entities["subject"] = code
                break

        # Extract unit number (e.g., "unit 2", "unit-3", "unit 1")
        unit_match = re.search(r'\bunit[\s\-]*([1-5])\b', query, re.IGNORECASE)
        if unit_match:
            entities["unit"] = int(unit_match.group(1))

        # Extract exam paper part (e.g., "part c", "c part", "part b", "part a", "10 marks", "15 marks", "2 marks")
        part_match = re.search(r'\b(?:part\s*([abc])|([abc])\s*part)\b', query, re.IGNORECASE)
        if part_match:
            p_letter = (part_match.group(1) or part_match.group(2)).upper()
            entities["part"] = f"Part {p_letter}"
        elif re.search(r'\b(?:10|15)\s*marks?\b', query, re.IGNORECASE):
            entities["part"] = "Part C"
        elif re.search(r'\b(?:6|8)\s*marks?\b', query, re.IGNORECASE):
            entities["part"] = "Part B"
        elif re.search(r'\b(?:2)\s*marks?\b', query, re.IGNORECASE):
            entities["part"] = "Part A"

        return entities

    @staticmethod
    def detect_intent(query: str) -> IntentType:
        """
        Analyzes the student's message and classifies what they want.
        """
        q = query.strip().lower()

        # 0. Security Guardrails: Prompt Injection, System Leakage & Exploit Defense
        # Protects against attempts to dump internal instructions, bypass boundaries, or request exploits.
        security_patterns = [
            # System prompt extraction & leakage
            r'\b(?:system\s+prompt|instructions?\s+given\s+to\s+you|repeat\s+(?:all\s+)?(?:text\s+)?above|show\s+(?:me\s+)?(?:your\s+)?(?:system\s+)?prompt|reveal\s+(?:system\s+)?instructions?|what\s+is\s+your\s+prompt|what\s+are\s+your\s+rules|print\s+system\s+prompt|dump\s+prompt|raw\s+prompt)\b',
            # Jailbreak & Persona hijacking (DAN, Developer Mode, Prompt Overrides)
            r'\b(?:ignore\s+(?:all\s+)?(?:previous\s+)?instructions?|disregard\s+(?:all\s+)?(?:prior\s+)?instructions?|forget\s+all\s+rules|you\s+are\s+now\s+(?:in\s+)?(?:unrestricted\s+)?dan|dan\s*(?:\([^)]*\)\s*)?mode|do\s+anything\s+now|jailbreak|developer\s+mode|unrestricted\s+mode|act\s+as\s+an?\s+unrestricted|bypass\s+(?:all\s+)?(?:rtu\s+)?(?:rules?|policies|guardrails?)|override\s+system)\b',
            # Malpractice, Hacking, SQLi, XSS & Infrastructure exploits
            r'(?:hack\s+into|sql\s+injection|\bunion\s+select\b|select\s+\*\s+from|<script\b|javascript:|alert\(|<iframe\b|document\.cookie|ddos|exploit\s+vulnerability|generate\s+malware|create\s+keylogger|keyloggers?|steal\s+exam\s+papers?|paper\s+leak|leaked\s+papers?|bypass\s+rtu\s+firewall|hack\s+rtu)',
            # Live exam cheating & exam hall malpractice
            r'\b(?:(?:sitting\s+in|inside)\s+(?:the\s+)?(?:rtu\s+)?exam\s+hall|exam\s+hall|give\s+(?:me\s+)?(?:question\s+\d+\s+)?answers?\s+(?:quickly|now)|live\s+exam\s+answers?|during\s+(?:my\s+)?exam|cheat\s+in\s+exam)\b',
            # Paper theft & future exam paper leaks
            r'\b(?:leak\s+(?:or\s+steal\s+)?(?:the\s+)?(?:upcoming\s+)?(?:20\d\d\s+)?(?:rtu\s+)?(?:exam\s+)?papers?|steal\s+.*?papers?|leak\s+.*?paper)\b',
            # Physical infrastructure sabotage
            r'\b(?:short-?circuit|sabotage|tamper\s+with\s+electrical|switchboard)\b',
            # Grade tampering
            r'\b(?:change\s+(?:my\s+)?(?:back\s+)?(?:paper\s+)?marks|tamper\s+with\s+grades|grade\s+tampering)\b',
            # Secret / Environment theft
            r'\b(?:dump\s+database\s+credentials|show\s+api\s+keys?|print\s+os\.environ|leak\s+secret|secret\s+key|database\s+password)\b'
        ]
        if any(re.search(pat, q) for pat in security_patterns):
            return IntentType.SECURITY_VIOLATION

        # 1. Greetings (e.g., "hi", "hello", "namaste", "how are you", "what's up", "how r u")
        if re.search(
            r'^\s*(?:'
            r'hi+|hey+|hello+|namaste|hola|yo|sup'
            r'|kaise\s+ho|kaisa\s+hai'
            r'|good\s+(?:morning|afternoon|evening)'
            r'|how\s+(?:r|are)\s+(?:u|you)(?:\s+doing)?'
            r"|what['\u2019]?s\s+up"
            r'|wassup|whatsup'
            r'|hru'
            r')(?:\s+(?:sir|senior|tutor|bhai|bro|ji|yaar))?[!?.\s]*$',
            q
        ):
            return IntentType.GREETING

        # 2. Closings & Courtesies (e.g., "thank you", "thanks bhai", "ok got it", "understood", "bye")
        if re.search(
            r'^\s*(?:(?:bhai|bro|sir|yaar)\s+)?(?:thanks?|thank\s+you|thx|shukriya|dhanyawa?d|ok|okay|got\s+it|understood|theek\s+hai|cool|alright|bye|goodbye|see\s+you|cya|nice|done|awesome)(?:\s+(?:bhai|bro|sir|yaar|thanks?|thank\s+you|got\s+it|understood|ok|okay))?[!?.\s]*$',
            q
        ):
            return IntentType.CLOSING

        # 3. Help & App Capabilities (e.g., "help from you", "can you help me", "what can you do", "features", "how to use")
        if re.search(
            r'\b(help|guide\s+me|what\s+can\s+you\s+do|how\s+to\s+use|features|about\s+you|who\s+are\s+you|your\s+role|capabilities|kya\s+kar\s+sakte|madad|commands|how\s+does\s+this\s+work|what\s+is\s+this\s+app)\b',
            q
        ):
            has_tech = any(re.search(rf'\b{re.escape(term)}\b', q) for term in TECH_TERMS)
            if not has_tech:
                return IntentType.HELP_CAPABILITIES

        # 4. Study Planning / Crash Course (e.g., "i have 6 hrs from now", "exam in 4 hours", "plan my revision")
        if re.search(
            r'\b(?:\d+(?:\.\d+)?\s*(?:hours?|hrs?|ghante|ghanta)|(?:time|hours?|hrs?)\s+left|prepare|plan|crash|roadmap|strategy|schedule|time\s+table)\b',
            q
        ):
            return IntentType.STUDY_PLAN

        # 5. Topic Importance / High Weightage (e.g., "most important topics in dsa", "repeated questions", "c part k liye important")
        if (
            re.search(
                r'\b(?:important\s+topics?|important\s+questions?|most\s+repeated|weightage|topic\s+ranking|high\s+yield|repeated\s+topics?|highest\s+marks|k(?:e|i)?\s+liye\s+important|important\s+(?:hai|batao|kya\s+hai|questions?|topics?)|sabse\s+important)\b',
                q
            )
            or re.search(r'\b(?:part\s*[abc]|[abc]\s*part)\b.*\b(?:important|weightage|repeated|questions?|topics?)\b', q)
            or re.search(r'\b(?:important|weightage|repeated)\b.*\b(?:part\s*[abc]|[abc]\s*part)\b', q)
        ):
            return IntentType.TOPIC_IMPORTANCE

        # 6. Past Year Questions (PYQs) (e.g., "past year questions", "pyq", "previous year questions", "past papers")
        if re.search(
            r'\b(past\s+(?:\d+\s+)?years?|previous\s+(?:\d+\s+)?years?|pyqs?|old\s+papers?|past\s+papers?|pichle\s+saal)\b',
            q
        ):
            has_tech = any(re.search(rf'\b{re.escape(term)}\b', q) for term in TECH_TERMS)
            if not has_tech:
                return IntentType.PAST_YEAR_QUESTIONS

        # 6b. BUG-12 FIX: Practice Quiz Request (e.g., "Quiz me with 3 questions on data structures.")
        if re.search(
            r'\b(quiz\s+me|quiz\s+on|test\s+me|practice\s+quiz|ask\s+me\s+(?:\d+\s+)?questions?|give\s+me\s+\d+\s+(?:practice\s+)?questions?)\b',
            q
        ):
            return IntentType.QUIZ_REQUEST

        # 7. Subject Selection / Declaration (e.g., "DSA", "dbms", "it is toc", "for dsa", "computer networks", "dsa subject hai")
        matched_subject = None
        matched_phrase = None
        for phrase, code in SUBJECT_ALIASES:
            if re.search(rf'\b{re.escape(phrase)}\b', q):
                matched_subject = code
                matched_phrase = phrase
                break

        if matched_subject:
            # Check if query is just declaring/stating a subject rather than asking a specific question
            has_tech = any(
                re.search(rf'\b{re.escape(term)}\b', q)
                for term in TECH_TERMS
                if term != matched_phrase and term not in matched_phrase
            )
            has_question_words = bool(re.search(
                r'\b(what|how|why|explain|define|solve|derive|calculate|difference|algorithm|diagram|trace|steps|unit)\b',
                q
            ))
            if not (has_tech or has_question_words):
                return IntentType.SUBJECT_SELECT

        # 8. General Out of Syllabus / Irrelevant (e.g., cricket, sports, politics, jokes, recipes, general trivia, medical, crypto)
        if re.search(
            r'\b(cricket|ipl|football|sports?|match|movie|song|poem|joke|recipe|food|dishes?|weather|prime\s+minister|president|capital\s+of|love|dating|girlfriend|boyfriend|bitcoin|crypto|stock\s+market|medical|fever|disease|treatment|jee\s+advanced|jee\s+main|iit\s+cutoff)\b',
            q
        ):
            return IntentType.OUT_OF_SYLLABUS

        # 9. Default to Exam Syllabus Question (triggers Hybrid FAISS + BM25 RAG)
        return IntentType.EXAM_QUESTION
