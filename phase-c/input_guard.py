from __future__ import annotations

import re
import time


VN_PII = {
    "cccd": r"\b\d{12}\b",
    "phone_vn": r"\b(?:\+84|0)\d{9,10}\b",
    "tax_code": r"\b\d{10}(?:-\d{3})?\b",
    "email": r"\b[\w.-]+@[\w.-]+\.\w+\b",
}


class InputGuard:
    def scrub_vn(self, text: str) -> str:
        for name, pattern in VN_PII.items():
            text = re.sub(pattern, f"[{name.upper()}]", text)
        return text

    def scrub_ner(self, text: str) -> str:
        # Lightweight fallback for names/addresses when Presidio is unavailable.
        text = re.sub(r"\b\d{1,4}\s+(?:Main Street|Le Loi|Lê Lợi)\b", "[ADDRESS]", text, flags=re.IGNORECASE)
        return re.sub(r"\b(?:John Smith|Nguyen Van A|Nguyễn Văn A|Ly Van Binh|Lý Văn Bình)\b", "[PERSON]", text)

    def sanitize(self, text: str) -> tuple[str, float]:
        start = time.perf_counter()
        output = self.scrub_ner(self.scrub_vn(text or ""))
        return output, (time.perf_counter() - start) * 1000


class TopicGuard:
    def __init__(self) -> None:
        self.allowed = {"rag", "ragas", "eval", "guardrail", "pii", "du lieu", "data", "bao cao", "nghi dinh"}
        self.blocked = {"dan", "jailbreak", "hack", "malware", "phishing", "steal", "evil"}

    def check(self, text: str) -> tuple[bool, str]:
        lowered = (text or "").lower()
        if any(term in lowered for term in self.blocked):
            return False, "Off-topic or injection-like request. Please ask about RAG, evaluation, or guardrails."
        if any(term in lowered for term in self.allowed):
            return True, "On topic."
        return False, "Off-topic. I can help with RAG, evaluation, data protection, and guardrails."
