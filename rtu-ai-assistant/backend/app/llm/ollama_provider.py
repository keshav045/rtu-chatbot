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
        # For lightweight local models (1.5B-3B), condense system prompt to prevent attention degradation and persona confusion
        concise_sys = (
            "You are the RTU Senior Academic Tutor and Exam Coach from Rajasthan Technical University. "
            "Explain engineering concepts directly to the junior student in a structured, high-scoring exam format: "
            "1. Technical Definition & Theoretical Basis\n"
            "2. Architecture / Diagram / Structure\n"
            "3. Step-by-Step Procedure, Algorithm, or Derivation\n"
            "4. Concrete Working Example / Code\n"
            "5. RTU Examiner Scoring Tips (how examiners award 8-15 marks).\n"
            "Speak directly as the Senior Tutor. NEVER output role labels such as 'Student Answer:' or simulate a student."
        )
        if "hinglish" in (system_prompt or "").lower():
            concise_sys += "\nReply in friendly, clear Hinglish (Hindi written in English script)."

        ctx_block = f"\n\n--- RTU PAST EXAM EVIDENCE ---\n{retrieved_context}" if (retrieved_context and retrieved_context.strip()) else ""

        messages = [
            {"role": "system", "content": f"{concise_sys}{ctx_block}"}
        ]

        if chat_history:
            for msg in chat_history[-4:]:
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

