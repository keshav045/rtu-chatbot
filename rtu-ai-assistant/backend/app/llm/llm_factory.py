"""
LLM Provider Factory for RTU AI Academic Assistant
Dynamically resolves and instantiates the appropriate LLM provider:
Ollama, Groq, Gemini, or OfflineMentor.
"""

import os
import httpx
from typing import Optional
from .base import BaseLLMProvider
from .ollama_provider import OllamaProvider
from .groq_provider import GroqProvider
from .gemini_provider import GeminiProvider
from .openai_provider import OpenAIProvider
from .offline_mentor_provider import OfflineMentorProvider

def is_ollama_online(base_url: str = "http://localhost:11434") -> bool:
    """Quick check if local Ollama daemon is active."""
    try:
        with httpx.Client(timeout=1.0) as client:
            res = client.get(f"{base_url}/api/tags")
            return res.status_code == 200
    except Exception:
        return False

def get_llm_provider(
    provider_type: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    ollama_url: str = "http://localhost:11434"
) -> BaseLLMProvider:
    """
    Factory resolving the LLM provider based on explicit parameter, environment variables,
    or automatic detection with graceful fallback to OfflineMentor.
    """
    choice = (provider_type or os.getenv("LLM_PROVIDER", "")).strip().lower()

    if choice == "openai":
        return OpenAIProvider(api_key=api_key or os.getenv("OPENAI_API_KEY"), model=model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))

    if choice == "groq":
        return GroqProvider(api_key=api_key or os.getenv("GROQ_API_KEY"), model=model_name or "llama-3.3-70b-versatile")

    if choice == "gemini":
        return GeminiProvider(api_key=api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"), model=model_name or "gemini-1.5-flash")

    if choice == "ollama":
        return OllamaProvider(base_url=ollama_url, model_name=model_name)

    if choice == "offline":
        return OfflineMentorProvider()

    # Automatic detection fallback chain
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))

    if os.getenv("GROQ_API_KEY"):
        return GroqProvider(api_key=os.getenv("GROQ_API_KEY"))

    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return GeminiProvider()

    if is_ollama_online(ollama_url):
        return OllamaProvider(base_url=ollama_url, model_name=model_name)

    # Safe deterministic offline mentor fallback
    return OfflineMentorProvider()

