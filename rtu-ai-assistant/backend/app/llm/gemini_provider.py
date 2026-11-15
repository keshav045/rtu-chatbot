"""
Google Gemini API Provider for RTU AI Academic Assistant
Supports Gemini 1.5 Flash and Gemini 1.5 Pro using google-generativeai.
"""

import os
from typing import List, Dict, Optional
from .base import BaseLLMProvider, DEFAULT_SYSTEM_PROMPT

class GeminiProvider(BaseLLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.model_name = model

    def generate(
        self,
        user_prompt: str,
        retrieved_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable.")

        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)

            full_system_instruction = (
                f"{system_prompt}\n\n"
                f"--- RETRIEVED RTU PAST QUESTIONS & SYLLABUS ---\n"
                f"{retrieved_context}"
            )

            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=full_system_instruction
            )

            # Build history if provided
            history = []
            if chat_history:
                for msg in chat_history[-6:]:
                    role = "user" if msg["role"] == "user" else "model"
                    history.append({"role": role, "parts": [msg["content"]]})

            chat = model.start_chat(history=history)
            response = chat.send_message(user_prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API generation failed: {e}")

