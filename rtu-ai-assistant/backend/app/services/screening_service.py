"""
Comprehensive Academic Screening & Guardrail Service for RTU AI Academic Assistant
Provides pre-generation and post-generation screening:
1. Input Screening: Scope validation, prompt injection defense, syllabus boundary enforcement.
2. Output Screening: Anti-leakage sanitization, hallucination screening, RTU 10-mark/2-mark rubric compliance.
3. Automated Evaluation Scoring: Generates quantitative evaluation metrics (0.0 - 1.0) for LangSmith tracing.
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Out-of-syllabus non-academic indicator keywords
OUT_OF_SCOPE_PATTERNS = [
    r"\b(cricket|ipl|score|football|fifa|messi|ronaldo|world\s*cup|kohli|dhoni)\b",
    r"\b(bollywood|hollywood|movie|cinema|actor|actress|trailer|box\s*office|netflix|youtube)\b",
    r"\b(recipe|cooking|cook|bake|baking|cake|biryani|paneer|pizza|burger|pasta|noodles|maggi|cocktail|coffee|tea|breakfast|lunch|dinner)\b",
    r"\b(weather|forecast|temperature\s+outside|rain\s+today|climate\s+today|flight|hotel|tourism)\b",
    r"\b(song|music|lyrics|poem|shayari|joke|memes?|pubg|bgmi|free\s*fire|gta)\b",
    r"\b(stock\s*market|crypto|bitcoin|trading|forex|lottery|betting)\b",
    r"\b(horoscope|astrology|zodiac|kundali)\b",
    r"\b(girlfriend|boyfriend|dating|love\s+advice|marriage|fever|medicine|doctor|weight\s+loss|diet\s+plan)\b",
    r"\b(politics|election|modi|rahul\s*gandhi|bjp|congress|vote|prime\s+minister|president|capital\s+of)\b",
    r"\b(jee\s+advanced|jee\s+main|iit\s+cutoff|neet)\b",
]

# Security & Prompt Injection attack patterns
INJECTION_PATTERNS = [
    r"(ignore|disregard|forget)\s+(all\s+)?(your\s+)?(previous|prior|above)?\s*instructions?",
    r"you\s+are\s+now\s+(in\s+)?(dan|developer|unrestricted|jailbreak)\s+mode",
    r"\b(act\s+as\s+an?\s+unrestricted|do\s+anything\s+now|jailbreak|developer\s+mode)\b",
    r"(reveal|print|dump|show)\s+(me\s+)?(your\s+)?(system\s+prompt|hidden\s+rules|rules|instructions?|api\s*keys?|credentials?)",
    r"repeat\s+(all\s+)?(everything|the\s+text)\s+(above|before)",
    r"bypass\s+(all\s+)?(content|security|safety|rtu)\s*(filters?|rules?|guardrails?)",
    r"leak\s+(the\s+)?(exam\s+paper|question\s+paper|upcoming\s+paper)",
    r"hack\s+(into\s+)?(rtu|university|portal|result|database)",
]

# Prompt Leakage Patterns in Output
LEAKAGE_PATTERNS = [
    r"You are Senior Tutor\s*[-—:]\s*an empathetic",
    r"Security & Anti-Leakage Rules:",
    r"Core Responsibilities:\s*1\.\s*Explain engineering",
    r"Under NO circumstances reveal, recite, or paraphrase",
    r"SENIOR_TUTOR_SYSTEM_PROMPT",
    r"\[REDACTED_API_KEY\]"
]


class InputScreeningResult(BaseModel):
    is_safe: bool = True
    is_in_syllabus: bool = True
    violation_reason: Optional[str] = None
    risk_level: str = "LOW"  # LOW, MEDIUM, HIGH
    scope_score: float = 1.0
    safety_score: float = 1.0
    suggested_redirect: Optional[str] = None


class OutputScreeningResult(BaseModel):
    is_safe: bool = True
    is_format_compliant: bool = True
    is_grounded: bool = True
    sanitized_output: str
    leakage_detected: bool = False
    format_score: float = 1.0
    groundedness_score: float = 1.0
    safety_score: float = 1.0
    warnings: List[str] = Field(default_factory=list)


class AcademicScreeningService:
    """
    Production screening guardrail validating inputs and outputs.
    Logs structured screening metrics for LangSmith evaluation.
    """

    @classmethod
    def screen_input(cls, query: str, active_subject: Optional[str] = None) -> InputScreeningResult:
        """
        Performs pre-generation screening on the student query.
        """
        q_lower = query.strip().lower()

        # 1. Check for prompt injection / security violations
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                return InputScreeningResult(
                    is_safe=False,
                    is_in_syllabus=False,
                    violation_reason="SECURITY_INJECTION_ATTEMPT",
                    risk_level="HIGH",
                    scope_score=0.0,
                    safety_score=0.0,
                    suggested_redirect=(
                        "I am the RTU Senior Academic Assistant. I operate strictly within the boundaries of the "
                        "Rajasthan Technical University (RTU) engineering syllabus. I cannot reveal internal system instructions, "
                        "bypass security rules, or assist with non-academic or unauthorized requests. "
                        "Please ask questions related to your RTU subjects, concepts, or exam preparation."
                    )
                )

        # 2. Check for obvious out-of-syllabus non-academic queries
        for pattern in OUT_OF_SCOPE_PATTERNS:
            if re.search(pattern, q_lower, re.IGNORECASE):
                # Ensure technical engineering homonyms aren't misflagged
                if not any(tech_w in q_lower for tech_w in ["algorithm", "complexity", "database", "circuit", "matrix", "network", "system", "frequency"]):
                    subj_hint = f"your current subject ({active_subject})" if active_subject else "RTU B.Tech subjects like DSA, DBMS, OS, or TOC"
                    return InputScreeningResult(
                        is_safe=True,
                        is_in_syllabus=False,
                        violation_reason="OUT_OF_SYLLABUS_TOPIC",
                        risk_level="MEDIUM",
                        scope_score=0.0,
                        safety_score=1.0,
                        suggested_redirect=(
                            f"Arre junior, yeh query RTU B.Tech ke academic syllabus ke bahar hai. "
                            f"Main sirf RTU exam preparation, concepts aur past year questions me help kar sakta hoon. "
                            f"Kripya {subj_hint} se related koi topic ya numerical poocho!"
                        )
                    )

        # Valid academic request
        return InputScreeningResult(
            is_safe=True,
            is_in_syllabus=True,
            risk_level="LOW",
            scope_score=1.0,
            safety_score=1.0
        )

    @classmethod
    def screen_output(
        cls,
        raw_output: str,
        query: str,
        retrieved_docs: Optional[List[Dict[str, Any]]] = None
    ) -> OutputScreeningResult:
        """
        Performs post-generation screening on the generated response.
        Enforces anti-leakage redaction, format checks, and groundedness checks.
        """
        warnings = []
        sanitized = raw_output or ""
        leakage = False
        safety_score = 1.0

        # 1. Anti-Leakage Screening
        for pattern in LEAKAGE_PATTERNS:
            if re.search(pattern, sanitized, re.IGNORECASE):
                leakage = True
                safety_score = 0.0
                sanitized = (
                    "I am the RTU Senior Academic Tutor. I operate strictly within the Rajasthan Technical University "
                    "curriculum and cannot reveal internal system instructions or configurations. Please ask questions "
                    "related to your RTU engineering subjects or exam preparation."
                )
                warnings.append("System prompt leak detected and intercepted.")
                break

        # Redact accidental API keys or secret strings
        sanitized = re.sub(
            r'\b(?:gsk_[A-Za-z0-9]{25,}|AIza[0-9A-Za-z-_]{35}|hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{30,})\b',
            '[REDACTED_API_KEY]',
            sanitized
        )

        # Strip emojis to maintain strict professional RTU senior tutor formatting
        sanitized = re.sub(r'[\U0001F300-\U0001FAFF\U00002600-\U000027BF]', '', sanitized)

        # 2. Format Compliance Screening
        format_score = 1.0
        q_lower = query.lower()
        is_10_mark = "10 mark" in q_lower or "part c" in q_lower or "15 mark" in q_lower
        is_2_mark = "2 mark" in q_lower or "part a" in q_lower

        if is_10_mark:
            has_def = any(k in sanitized.lower() for k in ["definition", "concept", "overview", "introduction"])
            has_steps_or_algo = any(k in sanitized.lower() for k in ["step", "algorithm", "mechanics", "working", "procedure"])
            has_tip = any(k in sanitized.lower() for k in ["tip", "scoring", "examiner", "marks", "mistake"])
            components_found = sum([has_def, has_steps_or_algo, has_tip])
            format_score = round(components_found / 3.0, 2)
            if format_score < 0.6:
                warnings.append("10-mark response lacks one or more standard RTU rubric components.")
        elif is_2_mark:
            lines = [line.strip() for line in sanitized.split("\n") if line.strip()]
            if len(lines) > 8:
                format_score = 0.7
                warnings.append("2-mark response is longer than ideal RTU concise format.")

        # 3. Groundedness / Citation Screening
        groundedness_score = 1.0
        if retrieved_docs:
            valid_years = {str(d.get("year")) for d in retrieved_docs if d.get("year")}
            mentioned_years = set(re.findall(r"\b(20[12]\d)\b", sanitized))
            # If model cites an exam year not present in retrieved context or syllabus
            fabricated_years = [y for y in mentioned_years if int(y) >= 2012 and int(y) <= 2026 and y not in valid_years]
            if len(fabricated_years) > 2:
                groundedness_score = 0.6
                warnings.append(f"Model mentioned ungrounded exam years: {fabricated_years}")

        return OutputScreeningResult(
            is_safe=not leakage,
            is_format_compliant=format_score >= 0.6,
            is_grounded=groundedness_score >= 0.6,
            sanitized_output=sanitized,
            leakage_detected=leakage,
            format_score=format_score,
            groundedness_score=groundedness_score,
            safety_score=safety_score,
            warnings=warnings
        )
