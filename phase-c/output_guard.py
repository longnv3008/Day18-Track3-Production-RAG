from __future__ import annotations

import os
import time


class OutputGuard:
    """Llama Guard 3 compatible interface.

    If GROQ_API_KEY is configured, this uses Groq's llama-guard-3-8b endpoint.
    Otherwise it falls back to a deterministic local safety check so the lab can
    still be run offline.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

    def check(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        if self.api_key:
            return self._check_groq(user_input, agent_response)
        return self._check_local(user_input, agent_response)

    def _check_groq(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        import requests

        start = time.perf_counter()
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={
                "model": "llama-guard-3-8b",
                "messages": [
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": agent_response},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()["choices"][0]["message"]["content"]
        latency_ms = (time.perf_counter() - start) * 1000
        is_safe = "safe" in result.lower() and "unsafe" not in result.lower()
        return is_safe, result, latency_ms

    def _check_local(self, user_input: str, agent_response: str) -> tuple[bool, str, float]:
        start = time.perf_counter()
        text = f"{user_input} {agent_response}".lower()
        unsafe_terms = ["hack", "malware", "steal", "jailbreak", "ignore all instructions", "leak cccd"]
        is_safe = not any(term in text for term in unsafe_terms)
        latency_ms = (time.perf_counter() - start) * 1000
        return is_safe, "safe" if is_safe else "unsafe", latency_ms
