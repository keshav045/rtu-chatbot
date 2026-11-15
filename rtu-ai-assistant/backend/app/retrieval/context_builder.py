"""
Context Builder for RAG Answer Generation
Prepares structured, token-budgeted context with physical PDF citations.
"""

from typing import List, Dict, Any

class ContextBuilder:
    def __init__(self, max_context_chars: int = 4000):
        self.max_context_chars = max_context_chars

    def build_context(self, retrieved_questions: List[Dict[str, Any]]) -> str:
        if not retrieved_questions:
            return "No matching past RTU examination questions found for the given criteria."

        blocks = []
        current_len = 0

        for i, q in enumerate(retrieved_questions, 1):
            block = (
                f"[Source {i} | Question ID: {q['question_id']}]\n"
                f"- Examination: RTU {q['year']} ({q['subject_short']}, {q['subject_code']}) | Semester {q['semester']}\n"
                f"- Section & Marks: {q['section']} ({q['marks']} Marks) | {q['question_number']}\n"
                f"- Source PDF: {q['source_pdf']} (Page {q['page_number']})\n"
                f"- Question Text: \"{q['question_text']}\"\n"
            )

            if current_len + len(block) > self.max_context_chars:
                break

            blocks.append(block)
            current_len += len(block)

        return "\n".join(blocks)

    def extract_citations(self, retrieved_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extracts structured citation metadata objects for API responses and PDF viewer navigation.
        """
        citations = []
        for q in retrieved_questions:
            citations.append({
                "question_id": q["question_id"],
                "subject": q["subject_short"],
                "subject_code": q["subject_code"],
                "year": q["year"],
                "semester": q["semester"],
                "section": q["section"],
                "question_number": q["question_number"],
                "marks": q["marks"],
                "source_pdf": q["source_pdf"],
                "page_number": q["page_number"],
                "question_text": q["question_text"]
            })
        return citations

