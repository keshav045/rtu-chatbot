"""
Groq API Provider for Fast Cloud Inference (llama-3.3-70b-versatile)
"""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider, DEFAULT_SYSTEM_PROMPT

class GroqProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model

    def generate(
        self,
        user_prompt: str,
        retrieved_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        if not self.api_key:
            raise ValueError("Groq API key not provided. Set GROQ_API_KEY environment variable or pass api_key.")

        from groq import Groq
        client = Groq(api_key=self.api_key)

        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n--- RETRIEVED RTU PAST QUESTIONS & SYLLABUS ---\n{retrieved_context}"}
        ]

        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1500
        )
        return completion.choices[0].message.content.strip()

