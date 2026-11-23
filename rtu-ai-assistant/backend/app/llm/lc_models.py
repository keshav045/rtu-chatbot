"""
LangChain LLM Provider Factory for RTU AI Academic Assistant
Provides unified ChatModel wrappers for Ollama, Groq, Gemini, and OpenAI.
"""

import os
from typing import Optional
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


def get_llm(provider: str, settings=None, model_name: Optional[str] = None):
    """
    Returns an instantiated LangChain ChatModel for the requested provider.
    Supported providers: 'ollama', 'groq', 'gemini', 'openai'.
    """
    p = (provider or "ollama").strip().lower()

    if settings is None:
        from app.core.config import settings as app_settings
        settings = app_settings

    if p == "ollama":
        return ChatOllama(
            model=model_name or settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            num_ctx=settings.OLLAMA_NUM_CTX,
            temperature=0.2
        )
    if p == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        return ChatGroq(
            model=model_name or settings.GROQ_MODEL,
            api_key=api_key,
            temperature=0.2
        )
    if p == "gemini":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=model_name or settings.GEMINI_MODEL,
            google_api_key=api_key,
            temperature=0.2
        )
    if p == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return ChatOpenAI(
            model=model_name or settings.OPENAI_MODEL,
            api_key=api_key,
            temperature=0.2
        )

    raise ValueError(f"Unknown provider: {provider}")
