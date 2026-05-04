"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


def _split_sentences(text: str) -> list[str]:
    """Split into sentences; supports Latin/VN punctuation and blank lines."""
    parts = re.split(r"(?<=[.!?])\s+|\n\s*\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load all markdown/text files from data/. (Đã implement sẵn)"""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Dùng TF–IDF + cosine giữa câu liền kề (không import sentence_transformers / transformers,
    tránh lỗi Keras 3 / TensorFlow trên một số môi trường).

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    metadata = metadata or {}
    sentences = _split_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        return [
            Chunk(
                text=sentences[0],
                metadata={**metadata, "chunk_index": 0, "strategy": "semantic"},
            )
        ]

    # char n-grams: on dinh hon voi tieng Viet (khong phu thuoc word boundary \b).
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True,
    )
    mat = vectorizer.fit_transform(sentences)

    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(cosine_similarity(mat[i - 1], mat[i])[0, 0])
        if sim < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    chunks: list[Chunk] = []
    for idx, grp in enumerate(groups):
        body = " ".join(grp).strip()
        if not body:
            continue
        chunks.append(
            Chunk(
                text=body,
                metadata={
                    **metadata,
                    "chunk_index": idx,
                    "strategy": "semantic",
                },
            )
        )
    return chunks


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    parents: list[Chunk] = []
    buf: list[str] = []
    cur_len = 0
    p_index = 0

    def _flush_parent() -> None:
        nonlocal buf, cur_len, p_index
        if not buf:
            return
        body = "\n\n".join(buf)
        pid = f"parent_{p_index}"
        parents.append(
            Chunk(
                text=body,
                metadata={
                    **metadata,
                    "chunk_type": "parent",
                    "parent_id": pid,
                },
                parent_id=None,
            )
        )
        p_index += 1
        buf = []
        cur_len = 0

    for para in paragraphs:
        add = len(para) + (2 if buf else 0)
        if buf and cur_len + add > parent_size:
            _flush_parent()
        buf.append(para)
        cur_len += add
    _flush_parent()

    if not parents and text.strip():
        pid = "parent_0"
        parents.append(
            Chunk(
                text=text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid},
                parent_id=None,
            )
        )

    children: list[Chunk] = []
    stride = max(1, child_size // 2)
    for p in parents:
        pid = p.metadata["parent_id"]
        pt = p.text
        if not pt.strip():
            continue
        c_idx = 0
        start = 0
        while start < len(pt):
            end = min(start + child_size, len(pt))
            piece = pt[start:end].strip()
            if piece:
                children.append(
                    Chunk(
                        text=piece,
                        metadata={
                            **metadata,
                            "chunk_type": "child",
                            "chunk_index": c_idx,
                            "parent_id": pid,
                        },
                        parent_id=pid,
                    )
                )
                c_idx += 1
            if end >= len(pt):
                break
            start += stride

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}
    headers = list(re.finditer(r"^#{1,3}\s+.+$", text, flags=re.MULTILINE))
    if not headers:
        t = text.strip()
        if not t:
            return []
        return [
            Chunk(
                text=t,
                metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0},
            )
        ]

    chunks: list[Chunk] = []
    for i, m in enumerate(headers):
        start = m.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section_text = text[start:end].strip()
        header_line = m.group(0).strip()
        if section_text:
            chunks.append(
                Chunk(
                    text=section_text,
                    metadata={
                        **metadata,
                        "section": header_line,
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    },
                )
            )
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.

    Returns:
        {"basic": {...}, "semantic": {...}, "hierarchical": {...}, "structure": {...}}
    """
    def _len_stats(chs: list[Chunk]) -> dict:
        if not chs:
            return {"num_chunks": 0, "avg_length": 0.0, "min_length": 0, "max_length": 0}
        lens = [len(c.text) for c in chs]
        return {
            "num_chunks": len(chs),
            "avg_length": sum(lens) / len(lens),
            "min_length": min(lens),
            "max_length": max(lens),
        }

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_struct: list[Chunk] = []
    total_parents = total_children = 0
    child_lens: list[int] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata") or {}
        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, threshold=SEMANTIC_THRESHOLD, metadata=meta))
        all_struct.extend(chunk_structure_aware(text, metadata=meta))
        par, ch = chunk_hierarchical(
            text,
            parent_size=HIERARCHICAL_PARENT_SIZE,
            child_size=HIERARCHICAL_CHILD_SIZE,
            metadata=meta,
        )
        total_parents += len(par)
        total_children += len(ch)
        child_lens.extend(len(c.text) for c in ch)

    hier_stats = {
        "num_parents": total_parents,
        "num_children": total_children,
        "num_chunks": total_children,
        "avg_length": (sum(child_lens) / len(child_lens)) if child_lens else 0.0,
        "min_length": min(child_lens) if child_lens else 0,
        "max_length": max(child_lens) if child_lens else 0,
    }

    return {
        "basic": _len_stats(all_basic),
        "semantic": _len_stats(all_semantic),
        "hierarchical": hier_stats,
        "structure": _len_stats(all_struct),
    }


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
