"""
Local Ollama LLM Provider for 8GB RAM Laptops
Interacts with local Ollama service (default: qwen2.5:1.5b or qwen2.5:3b).
"""

import httpx
from typing import List, Dict, Optional
from .base import BaseLLMProvider, DEFAULT_SYSTEM_PROMPT

class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "rtu-senior", timeout: float = 45.0):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        user_prompt: str,
        retrieved_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ) -> str:
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n--- RETRIEVED RTU PAST QUESTIONS & SYLLABUS ---\n{retrieved_context}"}
        ]

        if chat_history:
            for msg in chat_history[-6:]:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_prompt})

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model_name,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "top_p": 0.9,
                            "num_predict": 1024
                        }
                    }
                )
                if res.status_code == 200:
                    data = res.json()
                    return data.get("message", {}).get("content", "").strip()
                else:
                    return f"Ollama returned error status: {res.status_code}"
        except Exception as e:
            raise RuntimeError(f"Ollama connection error at {self.base_url}: {e}")

