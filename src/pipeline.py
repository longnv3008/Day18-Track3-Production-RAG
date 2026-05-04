"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY


# ─── Latency tracker ────────────────────────────────────


class LatencyTracker:
    """Accumulate per-step latency for the bonus latency breakdown report."""

    def __init__(self):
        self._data: dict[str, list[float]] = {}

    def record(self, step: str, elapsed_ms: float) -> None:
        self._data.setdefault(step, []).append(elapsed_ms)

    def summary(self) -> dict[str, dict]:
        return {
            step: {
                "calls": len(times),
                "total_ms": round(sum(times), 1),
                "avg_ms": round(sum(times) / len(times), 1),
                "min_ms": round(min(times), 1),
                "max_ms": round(max(times), 1),
            }
            for step, times in self._data.items()
        }


_latency = LatencyTracker()


# ─── LLM Generation ─────────────────────────────────────


def _generate_answer(query: str, contexts: list[str]) -> str:
    """
    Generate a grounded answer using OpenAI gpt-4o-mini.
    Falls back to returning the best retrieved context if API is unavailable.
    """
    if not OPENAI_API_KEY:
        return contexts[0] if contexts else "Không tìm thấy thông tin."

    try:
        from openai import OpenAI
        t0 = time.perf_counter()
        client = OpenAI(api_key=OPENAI_API_KEY)
        context_str = "\n\n---\n\n".join(contexts[:3])
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý tra cứu tài liệu. "
                        "Trả lời DỰA TRÊN context được cung cấp. "
                        "Nếu context không có thông tin chính xác về câu hỏi, "
                        "hãy tóm tắt những gì context CÓ liên quan và nêu rõ giới hạn. "
                        "KHÔNG bịa đặt số liệu. Luôn trích dẫn từ context. "
                        "Trả lời ngắn gọn, súc tích bằng tiếng Việt."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                },
            ],
            temperature=0.0,
            max_tokens=512,
        )
        _latency.record("llm_generate", (time.perf_counter() - t0) * 1000)
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  [WARN] LLM generation failed ({exc}). Using fallback context.")
        return contexts[0] if contexts else "Không tìm thấy thông tin."


# ─── Pipeline build ──────────────────────────────────────


def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents...")
    t0 = time.perf_counter()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })
    chunk_ms = (time.perf_counter() - t0) * 1000
    _latency.record("chunking", chunk_ms)
    print(f"  {len(all_chunks)} chunks from {len(docs)} documents  ({chunk_ms:.0f} ms)")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    t0 = time.perf_counter()
    enriched = enrich_chunks(all_chunks, methods=["contextual", "hyqa", "metadata"])
    enrich_ms = (time.perf_counter() - t0) * 1000
    _latency.record("enrichment", enrich_ms)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  Enriched {len(enriched)} chunks  ({enrich_ms:.0f} ms)")
    else:
        print(f"  [WARN] M5 fallback -- using raw chunks  ({enrich_ms:.0f} ms)")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    t0 = time.perf_counter()
    search = HybridSearch()
    search.index(all_chunks)
    index_ms = (time.perf_counter() - t0) * 1000
    _latency.record("indexing", index_ms)
    print(f"  Indexed {len(all_chunks)} chunks  ({index_ms:.0f} ms)")

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker...")
    t0 = time.perf_counter()
    reranker = CrossEncoderReranker()
    rerank_load_ms = (time.perf_counter() - t0) * 1000
    _latency.record("reranker_load", rerank_load_ms)
    print(f"  Reranker ready  ({rerank_load_ms:.0f} ms)")

    return search, reranker


# ─── Single-query runner ─────────────────────────────────


def run_query(
    query: str, search: HybridSearch, reranker: CrossEncoderReranker
) -> tuple[str, list[str]]:
    """Run a single query through the full pipeline and return (answer, contexts)."""
    # Hybrid search
    t0 = time.perf_counter()
    results = search.search(query)
    _latency.record("search", (time.perf_counter() - t0) * 1000)

    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]

    # Rerank
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    _latency.record("rerank", (time.perf_counter() - t0) * 1000)

    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    # LLM generation
    answer = _generate_answer(query, contexts)
    return answer, contexts


# ─── Evaluation runner ───────────────────────────────────


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker) -> dict:
    """Run RAGAS evaluation over the full test set."""
    print("\n[Eval] Running queries...")
    test_set = load_test_set()
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:55]}...")

    print("\n[Eval] Running RAGAS...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'[PASS]' if s >= 0.75 else '[FAIL]'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)

    # Bonus: latency breakdown
    lat = _latency.summary()
    print("\n[Latency] Breakdown per step:")
    for step, info in lat.items():
        print(f"  {step:<18} avg={info['avg_ms']:>7.1f} ms  total={info['total_ms']:>8.1f} ms")

    # Persist latency info alongside main report
    lat_path = "latency_report.json"
    with open(lat_path, "w", encoding="utf-8") as f:
        json.dump(lat, f, ensure_ascii=False, indent=2)
    print(f"  Latency report → {lat_path}")

    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
