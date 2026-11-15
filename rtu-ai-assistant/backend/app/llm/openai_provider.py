"""
OpenAI API Provider for RTU AI Academic Assistant (GPT-4o-mini / GPT-4o)
Uses httpx for robust, lightweight, direct API execution.
"""

import os
import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider, DEFAULT_SYSTEM_PROMPT

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate(
        self,
        user_prompt: str,
        retrieved_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable or pass api_key.")

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n--- RETRIEVED RTU PAST QUESTIONS & SYLLABUS ---\n{retrieved_context}"}
        ]

        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1500
        }

        with httpx.Client(timeout=45.0) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            if resp.status_code != 200:
                raise RuntimeError(f"OpenAI API Error {resp.status_code}: {resp.text}")
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
