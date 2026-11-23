"""
LangChain LCEL Answer Chain for RTU AI Academic Assistant
Constructs an end-to-end Runnable chain combining document retrieval,
formatting, prompt templating, LLM synthesis, and string output parsing.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

SENIOR_TUTOR_SYSTEM_PROMPT = """You are Senior Tutor - an empathetic, high-scoring senior student from Rajasthan Technical University (RTU).
Answer in the exam-oriented RTU format (10-mark Part C / 2-mark Part A) using only the past-paper context below and cite it.

Formatting Guidelines:
- For 10-Mark (Part C) Questions:
  1. Formal Definition & Concept Summary (2 Marks)
  2. Labeled Diagram / Architecture Flow Guide (3 Marks)
  3. Step-by-Step Algorithm or Mathematical Working (3 Marks)
  4. Concrete Example / Traced Values (1 Mark)
  5. RTU Examiner Scoring Tip & Pitfalls to Avoid (1 Mark)
- For 2-Mark (Part A) Questions: Crisp 2-3 line definition + one formula/property.

Context:
{context}"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SENIOR_TUTOR_SYSTEM_PROMPT),
    ("human", "{question}"),
])


def format_docs(docs: List[Document]) -> str:
    """Formats retrieved LangChain documents into context with citation headers."""
    formatted_chunks = []
    for d in docs:
        subj = d.metadata.get("subject", "RTU")
        yr = d.metadata.get("year", "")
        sec = d.metadata.get("section", "")
        marks = d.metadata.get("marks", "")
        formatted_chunks.append(
            f"[{subj} {yr} {sec} {marks}m] {d.page_content}"
        )
    return "\n\n".join(formatted_chunks)


def build_chain(retriever, llm, prompt_template: Optional[ChatPromptTemplate] = None):
    """
    Builds the LCEL chain:
    ({'context': retriever | format_docs, 'question': RunnablePassthrough()}
     | PROMPT | llm | StrOutputParser())
    """
    selected_prompt = prompt_template or PROMPT
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | selected_prompt
        | llm
        | StrOutputParser()
    )
