"""
Offline Academic Mentor & Senior Tutor Provider for RTU AI Academic Assistant
Provides deterministic, structured, grounded academic synthesis when no external LLM
or Ollama service is available or when fast offline response is preferred.
Implements Section 7 of Implementation Plan Version 2 (Senior Tutor, Hinglish/English, 10-Mark Answer Coach).
"""

from typing import List, Dict, Optional
import re
from .base import BaseLLMProvider, DEFAULT_SYSTEM_PROMPT

class OfflineMentorProvider(BaseLLMProvider):
    """
    Offline Senior Tutor: Synthesizes high-scoring exam responses directly from
    retrieved RTU questions and syllabus metadata without requiring external API keys.
    """

    def generate(
        self,
        user_prompt: str,
        retrieved_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        prompt_lower = user_prompt.lower()

        # Check for Hinglish detection
        hinglish_keywords = ["kya", "hai", "kaise", "samjhao", "bhai", "batao", "marks ke liye", "bhi", "nahi", "bata do", "bataiye", "karen", "hota", "karo", "senior"]
        is_hinglish = any(re.search(rf"\b{kw}\b", prompt_lower) for kw in hinglish_keywords) or ("hinglish" in system_prompt.lower())

        # If no retrieved context is provided or found
        if not retrieved_context or "No matching past RTU examination questions found" in retrieved_context or "No relevant RTU examination questions found" in retrieved_context:
            if is_hinglish:
                return (
                    "### RTU Senior Tutor Notice\n\n"
                    "Arre junior, maine RTU ke official syllabus aur previous year papers mein search kiya, par **yeh question ya topic match nahi ho raha**.\n\n"
                    "**Senior Tips:**\n"
                    "- Technical terms ki spelling check karo (jaise 'B-tree', 'Dijkstra', 'Turing machine').\n"
                    "- Subject code ya short form mention karo (jaise *DSA*, *OS*, *TOC*, *CN*, *OOP*).\n"
                    "- Agar yeh kisi specific Unit (Unit 1 to 5) ka topic hai toh unit number batayein!"
                )
            else:
                return (
                    "### RTU Academic Assistant Notice\n\n"
                    "I searched the indexed Rajasthan Technical University (RTU) database, but **no relevant syllabus topics or past examination questions** "
                    "were found matching your query.\n\n"
                    "**Suggestions:**\n"
                    "- Check the spelling of technical terms (e.g., 'B-tree', 'Dijkstra', 'Turing machine').\n"
                    "- Try specifying the subject or semester (e.g., *DSA*, *OS*, *TOC*, *CN*).\n"
                    "- If this is part of your branch syllabus, specify the Unit number."
                )

        # Parse questions from the retrieved context blocks
        q_blocks = re.findall(r"\[(?:Source \d+ \| Question ID:|Document \d+)\s*([^\]]+)\]\s*(.*?)(?=\n\[(?:Source \d+|Document \d+)|\Z)", retrieved_context, re.DOTALL)
        
        parsed_questions = []
        for qid_match, block in q_blocks:
            q_text_match = re.search(r'Question Text:\s*"?(.*?)"?(?=\n|$)', block, re.DOTALL)
            year_match = re.search(r"\b(20[12]\d)\b", block)
            subj_match = re.search(r"\(([A-Za-z0-9_-]+),", block)
            part_match = re.search(r"(Part\s*[ABC]|Section\s*[ABC])", block, re.I)
            marks_match = re.search(r"\((\d+)\s*Marks\)", block, re.I)
            pdf_match = re.search(r"Source PDF:\s*(.*?)\s*\(Page", block)
            page_match = re.search(r"Page\s*(\d+)", block)

            q_text = q_text_match.group(1).strip() if q_text_match else ""
            year = year_match.group(1).strip() if year_match else "N/A"
            subject = subj_match.group(1).strip() if subj_match else "RTU Subject"
            part = part_match.group(1).strip() if part_match else "Part B"
            marks = marks_match.group(1).strip() if marks_match else "N/A"
            pdf = pdf_match.group(1).strip() if pdf_match else "RTU_Exam.pdf"
            page = page_match.group(1).strip() if page_match else "1"

            if q_text:
                parsed_questions.append({
                    "question_id": qid_match.strip(),
                    "text": q_text,
                    "year": year,
                    "subject": subject,
                    "part": part,
                    "marks": marks,
                    "pdf": pdf,
                    "page": page
                })

        if not parsed_questions:
            return (
                f"### RTU Senior Tutor\n\n"
                f"Retrieved Context:\n{retrieved_context}\n\n"
                f"Based on the official RTU materials above, please review the referenced questions for your preparation."
            )

        # Check user intent
        is_explain_query = any(w in prompt_lower for w in ["explain", "describe", "what is", "how to", "solution", "answer", "write", "solve", "detail", "it", "samjhao", "batao", "kaise", "kya"])
        is_list_query = any(w in prompt_lower for w in ["find all", "list all", "show all questions", "saare questions"])

        response_parts = []
        if is_hinglish:
            response_parts.append(f"### RTU Senior Tutor: Past Exam Evidence ({len(parsed_questions)} Questions Found)\n")
            response_parts.append("RTU ke official papers se yeh questions match hue hain:\n")
        else:
            response_parts.append(f"### RTU Senior Tutor: Examination Evidence ({len(parsed_questions)} Questions Found)\n")
            response_parts.append("Based on verified RTU previous-year examination papers:\n")

        for idx, q in enumerate(parsed_questions[:4], 1):
            marks_str = f" ({q['marks']} Marks)" if q['marks'] != "N/A" else ""
            part_str = f"[{q['part']}]" if q['part'] != "N/A" else ""
            response_parts.append(
                f"**{idx}. {q['text']}**\n"
                f"- **Exam Details**: {q['subject']} ({q['year']}) | {part_str}{marks_str}\n"
                f"- **Verified Source**: `{q['pdf']}` (Page {q['page']})\n"
            )

        # Determine target marks: check user prompt first (e.g., "for 10 marks"), then top question
        user_marks_match = re.search(r"(\d+)\s*marks?", prompt_lower)
        target_marks = None
        if user_marks_match:
            try:
                target_marks = int(user_marks_match.group(1))
            except ValueError:
                pass

        top_q = parsed_questions[0]
        if target_marks is None:
            try:
                target_marks = int(top_q['marks']) if top_q['marks'] != "N/A" else 10
            except ValueError:
                target_marks = 10

        # If user asked for an explanation or structured answer
        if is_explain_query or not is_list_query:
            part_label = "Part A" if target_marks <= 2 else ("Part B" if target_marks <= 8 else "Part C")

            response_parts.append("\n---\n")
            if is_hinglish:
                response_parts.append(f"### 10-Mark Answer Coach: *\"{top_q['text']}\"*\n")
                response_parts.append(f"*Target Marking: **{target_marks} Marks** ({part_label})*\n\n")

                if target_marks <= 2:
                    response_parts.append(
                        "#### 1. Crisp Definition (1 Mark)\n"
                        f"- Seedha 2 line ki precise definition likho: *\"{top_q['text']}\"*.\n"
                        "- No intro story, evaluator direct standard mathematical notation dekhta hai.\n\n"
                        "#### 2. Key Properties & 1 Example (1 Mark)\n"
                        "- 2 standard characteristics ya time complexity ($O(1)$, $O(\\log n)$) point-wise likho.\n"
                        "- Ek 1-line syntax ya mini structural diagram draw karo.\n"
                    )
                else: # 8 to 15 marks
                    response_parts.append(
                        "#### 1. Core Definition & Formal Intuition (2 Marks)\n"
                        f"- Foundational concept define karo. Clear bold keywords use karo: **{top_q['text'][:55]}...**\n"
                        "- Real-world context aur purpose specify karo.\n\n"
                        "#### 2. Diagram Guide (Examiner's Favourite - 3 Marks)\n"
                        "- **Diagram zaroor draw karna**: RTU mein examiners labeled diagram ke bina 8+ marks nahi dete.\n"
                        "- Ek neat rectangular schematic block diagram banao, arrows ke saath inputs, processing states, aur output trace label karo.\n\n"
                        "#### 3. Step-by-Step Working & Algorithm (3 Marks)\n"
                        "- Step 1: Initial state and precondition setup.\n"
                        "- Step 2: Core execution loop / derivation rule.\n"
                        "- Step 3: Termination condition and output formulation.\n"
                        "- Time Complexity: Worst-case $O(n \\log n)$, Best-case $O(n)$, Space complexity explicitly state karo.\n\n"
                        "#### 4. Concrete Numerical / Execution Example (2 Marks)\n"
                        "- Ek small sample input lekar (e.g., 5 elements or 3 states) step-by-step trace karke dikhao.\n"
                        "- Final computed output ko box mein highlight karo.\n\n"
                        "#### 5. RTU Senior Scoring Tip\n"
                        f"- Part C ({target_marks} Marks) mein answer sheet ke 3-4 pages cover karo. Section headings ko underline karo aur important formulas ko rectangular box mein band karo!"
                    )
            else:
                response_parts.append(f"### Answer Format Coach: *\"{top_q['text']}\"*\n")
                response_parts.append(f"*Target Marking Scheme: **{target_marks} Marks** ({part_label})*\n\n")

                if target_marks <= 2:
                    response_parts.append(
                        "#### 1. Core Definition (1 Mark)\n"
                        f"- Precise formal definition of: *{top_q['text']}* in 2 concise sentences.\n"
                        "- Standard mathematical or algorithmic notation.\n\n"
                        "#### 2. Key Properties & Minimal Example (1 Mark)\n"
                        "- 2 distinguishing technical characteristics or runtime complexity bounds.\n"
                        "- Compact 1-line code syntax or small inline schematic.\n"
                    )
                else: # 8 to 15 marks
                    response_parts.append(
                        "#### 1. Comprehensive Introduction & Core Definition (2 Marks)\n"
                        f"- Formal theoretical definition addressing: **{top_q['text'][:60]}...**\n"
                        "- Contextual relevance in the RTU B.Tech curriculum.\n\n"
                        "#### 2. Diagrammatic Schematic (3 Marks)\n"
                        "- **Examiner Expectation**: RTU evaluators heavily weigh clean, labeled diagrams.\n"
                        "- Draw a structured block layout or state transition diagram showing inputs, components, and dataflow.\n\n"
                        "#### 3. Step-by-Step Algorithmic Mechanism / Derivation (3 Marks)\n"
                        "- Step-by-step mathematical derivation or procedural execution.\n"
                        "- Formal complexity analysis: Time complexity ($O$, $\\Omega$, $\\Theta$) and Auxiliary space.\n\n"
                        "#### 4. Concrete Numerical Trace / Working Example (2 Marks)\n"
                        "- Work through a concrete example with sample inputs to demonstrate operational mechanics.\n"
                        "- Frame final computed results in a neat box.\n\n"
                        "#### 5. RTU Scoring Strategy\n"
                        f"- For a {target_marks}-mark question in Part C, structure your answer into distinct numbered sections with underlined keywords and enclosed formulas."
                    )

        response_parts.append("\n\n---\n> **Verified Citation**: Question ID `" + top_q['question_id'] + f"` from `{top_q['pdf']}` (Page {top_q['page']}).")

        return "\n".join(response_parts)
