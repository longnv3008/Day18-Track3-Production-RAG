"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Vietnamese keyword helpers ───────────────────────────

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("finance",   ["thuế", "doanh thu", "lợi nhuận", "gtgt", "tài chính", "kế toán",
                   "thue", "doanh thu", "loi nhuan", "tai chinh", "ke toan"]),
    ("policy",    ["nghị định", "quy định", "điều", "khoản", "pháp luật", "bộ luật",
                   "nghi dinh", "quy dinh", "dieu", "khoan", "phap luat"]),
    ("hr",        ["nhân viên", "nghỉ phép", "lương", "hợp đồng lao động",
                   "nhan vien", "nghi phep", "luong", "hop dong"]),
    ("data",      ["dữ liệu cá nhân", "bảo vệ dữ liệu", "xử lý dữ liệu",
                   "du lieu", "bao ve", "xu ly"]),
    ("it",        ["phần mềm", "hệ thống", "mạng", "bảo mật", "máy chủ",
                   "phan mem", "he thong", "mang", "bao mat"]),
]


def _detect_category(text: str) -> str:
    text_lower = text.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in text_lower for kw in keywords):
            return category
    return "general"


def _extract_entities(text: str) -> list[str]:
    """Heuristic: capitalised phrases and numeric values as entities."""
    entities = []
    # Numbers with units
    entities += re.findall(r"\d[\d.,]*\s*(?:ty|trieu|nghin|ngay|gio|%|VND|USD|đồng)", text, re.IGNORECASE)
    # Proper-noun-like runs (2-4 ALLCAPS/Title words)
    entities += re.findall(r"(?:[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zà-ỹ]+\s+){1,3}[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zà-ỹ]+", text)
    return list(dict.fromkeys(e.strip() for e in entities if e.strip()))[:6]


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.

    Thử OpenAI nếu có API key; fallback sang extractive (2 câu đầu).
    """
    # ── Option A: OpenAI ──────────────────────────────────
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            pass

    # ── Option B: Extractive (2 câu đầu) ─────────────────
    sentences = re.split(r"(?<=[.!?。])\s+|\n{2,}", text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    return ". ".join(sentences[:2]).rstrip(".") + "." if sentences else text[:200]


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.

    Thử OpenAI nếu có API key; fallback sang template-based generation.
    """
    # ── Option A: OpenAI ──────────────────────────────────
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên 1 dòng, không đánh số."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            lines = resp.choices[0].message.content.strip().split("\n")
            return [q.strip().lstrip("0123456789.-) ") for q in lines if q.strip()][:n_questions]
        except Exception:
            pass

    # ── Option B: Template-based ──────────────────────────
    # Extract key facts and wrap with question templates
    numbers = re.findall(r"\d[\d.,]*\s*(?:\w+)?", text)
    category = _detect_category(text)

    templates_by_cat: dict[str, list[str]] = {
        "finance": [
            "Giá trị {} là bao nhiêu?",
            "Mức thuế áp dụng là bao nhiêu?",
            "Doanh thu trong kỳ là bao nhiêu?",
        ],
        "policy": [
            "Điều khoản này quy định về vấn đề gì?",
            "Nghĩa vụ của tổ chức theo quy định này là gì?",
            "Điều kiện để áp dụng quy định này là gì?",
        ],
        "data": [
            "Điều kiện chuyển giao dữ liệu cá nhân là gì?",
            "Tổ chức phải thực hiện nghĩa vụ gì khi xử lý dữ liệu?",
            "Trong bao lâu phải báo cáo sự cố lộ dữ liệu?",
        ],
        "hr": [
            "Nhân viên được hưởng quyền lợi gì?",
            "Điều kiện để được nghỉ phép là gì?",
            "Mức lương quy định là bao nhiêu?",
        ],
        "general": [
            "Nội dung chính của đoạn này là gì?",
            "Đoạn văn đề cập đến vấn đề gì?",
            "Thông tin quan trọng trong đoạn này là gì?",
        ],
    }

    templates = templates_by_cat.get(category, templates_by_cat["general"])
    questions = []
    for tmpl in templates[:n_questions]:
        if "{}" in tmpl and numbers:
            questions.append(tmpl.format(numbers[0]))
        else:
            questions.append(tmpl)
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.

    Thử OpenAI nếu có API key; fallback sang rule-based prepend.
    Anthropic benchmark: giảm 49% retrieval failure.
    """
    # ── Option A: OpenAI ──────────────────────────────────
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu.",
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=80,
            )
            context_sentence = resp.choices[0].message.content.strip()
            return f"{context_sentence}\n\n{text}"
        except Exception:
            pass

    # ── Option B: Rule-based prepend ─────────────────────
    category = _detect_category(text)
    category_labels = {
        "finance": "thông tin tài chính",
        "policy": "quy định pháp lý",
        "data": "bảo vệ dữ liệu cá nhân",
        "hr": "chính sách nhân sự",
        "it": "thông tin công nghệ",
        "general": "nội dung tổng quát",
    }
    label = category_labels.get(category, "nội dung tài liệu")
    prefix = f"[Trích từ: {document_title}] Đoạn này thuộc chủ đề {label}." if document_title else f"Đoạn này thuộc chủ đề {label}."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    Trích xuất metadata tự động: topic, entities, date_range, category.

    Thử OpenAI nếu có API key; fallback sang rule-based extraction.
    """
    # ── Option A: OpenAI ──────────────────────────────────
    if OPENAI_API_KEY:
        try:
            import json as _json
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance|data|general", "language": "vi|en"}',
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")
            return _json.loads(raw)
        except Exception:
            pass

    # ── Option B: Rule-based ──────────────────────────────
    category = _detect_category(text)
    entities = _extract_entities(text)

    # Detect language (basic heuristic: Vietnamese diacritics)
    vi_chars = len(re.findall(r"[àáâãèéêìíòóôõùúăđĩũơưạảấầẩẫậắằẳẵặẹẻẽềềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", text.lower()))
    language = "vi" if vi_chars > 5 else "en"

    # Topic: first meaningful noun phrase (simplified)
    words = text.split()
    topic = " ".join(words[:5]).rstrip(".,;:") if words else "unknown"

    return {
        "topic": topic,
        "entities": entities,
        "category": category,
        "language": language,
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    apply_summary = "summary" in methods or "full" in methods
    apply_hyqa = "hyqa" in methods or "full" in methods
    apply_contextual = "contextual" in methods or "full" in methods
    apply_metadata = "metadata" in methods or "full" in methods

    enriched: list[EnrichedChunk] = []

    for chunk in chunks:
        raw_text = chunk.get("text", "")
        meta = chunk.get("metadata", {})
        source = meta.get("source", "")

        summary = summarize_chunk(raw_text) if apply_summary else ""
        questions = generate_hypothesis_questions(raw_text) if apply_hyqa else []
        enriched_text = contextual_prepend(raw_text, source) if apply_contextual else raw_text
        auto_meta = extract_metadata(raw_text) if apply_metadata else {}

        # Merge HyQA questions into enriched text to bridge vocabulary gap
        if questions and apply_hyqa:
            q_block = "\n".join(f"Q: {q}" for q in questions)
            enriched_text = f"{enriched_text}\n\n{q_block}"

        enriched.append(EnrichedChunk(
            original_text=raw_text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**meta, **auto_meta},
            method="+".join(methods),
        ))

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
