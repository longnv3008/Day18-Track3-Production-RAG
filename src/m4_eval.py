"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os
import re
import sys
import json
from dataclasses import dataclass
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


# ─── Vietnamese-aware tokeniser ──────────────────────────

_VI_STOPWORDS = {
    "là", "của", "và", "trong", "có", "được", "các", "này", "đó", "với",
    "để", "từ", "một", "không", "theo", "về", "như", "thì", "hay", "hoặc",
    "khi", "nếu", "vì", "tại", "do", "bởi", "cho", "trên", "dưới", "bao",
    "nhiêu", "gì", "nào", "ai", "đâu", "sao", "bị", "ra", "vào", "đã",
    "sẽ", "đang", "rằng", "mà", "thế", "thi", "cũng", "lại", "còn",
    # Latin stopwords (for non-accented text)
    "la", "cua", "va", "trong", "co", "duoc", "cac", "nay", "do", "voi",
    "de", "tu", "mot", "khong", "theo", "ve", "nhu", "thi", "hay", "hoac",
    "khi", "neu", "vi", "tai", "do", "boi", "cho", "tren", "duoi",
    "bao", "nhieu", "gi", "nao", "ai", "dau", "sao", "bi", "ra", "vao",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, strip Vietnamese stopwords."""
    tokens = re.findall(r"[a-zA-ZÀ-ỹà-ỹ0-9]+", text.lower())
    return [t for t in tokens if t not in _VI_STOPWORDS and len(t) > 1]


def _overlap_ratio(source_tokens: list[str], target_text: str) -> float:
    """Fraction of source_tokens that appear in target_text."""
    if not source_tokens:
        return 0.0
    target_lower = target_text.lower()
    hits = sum(1 for tok in source_tokens if tok in target_lower)
    return hits / len(source_tokens)


# ─── Heuristic fallback metrics ─────────────────────────


def _heuristic_faithfulness(answer: str, contexts: list[str]) -> float:
    """Answer tokens should be grounded in the contexts."""
    ctx_text = " ".join(contexts)
    tokens = _tokenize(answer)
    score = _overlap_ratio(tokens, ctx_text)
    # Penalise very short answers (< 5 tokens) slightly
    if len(tokens) < 5:
        score = max(score * 0.9, 0.0)
    return round(min(score, 1.0), 4)


def _heuristic_answer_relevancy(question: str, answer: str) -> float:
    """Answer should address key terms in the question."""
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    score = _overlap_ratio(q_tokens, answer)
    return round(min(score, 1.0), 4)


def _heuristic_context_precision(question: str, contexts: list[str]) -> float:
    """Proportion of retrieved contexts that are relevant to the question."""
    if not contexts:
        return 0.0
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0
    relevant = sum(
        1 for ctx in contexts if _overlap_ratio(q_tokens[:8], ctx) > 0.1
    )
    return round(relevant / len(contexts), 4)


def _heuristic_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Coverage of ground-truth information in the retrieved contexts."""
    ctx_text = " ".join(contexts)
    gt_tokens = _tokenize(ground_truth)
    score = _overlap_ratio(gt_tokens, ctx_text)
    return round(min(score, 1.0), 4)


def _heuristic_evaluate(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """Rule-based RAGAS-style evaluation (no LLM required)."""
    per_question: list[EvalResult] = []
    for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths):
        f = _heuristic_faithfulness(a, ctx)
        ar = _heuristic_answer_relevancy(q, a)
        cp = _heuristic_context_precision(q, ctx)
        cr = _heuristic_context_recall(gt, ctx)
        per_question.append(EvalResult(q, a, ctx, gt, f, ar, cp, cr))

    return {
        "faithfulness": round(mean(r.faithfulness for r in per_question), 4),
        "answer_relevancy": round(mean(r.answer_relevancy for r in per_question), 4),
        "context_precision": round(mean(r.context_precision for r in per_question), 4),
        "context_recall": round(mean(r.context_recall for r in per_question), 4),
        "per_question": per_question,
    }


# ─── Main public API ────────────────────────────────────


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    """
    Run RAGAS evaluation on provided Q/A/context/ground-truth tuples.

    Tries the official ragas library first (requires OpenAI API key).
    Falls back to heuristic metrics when ragas is unavailable or fails.

    Returns:
        {
          "faithfulness": float,
          "answer_relevancy": float,
          "context_precision": float,
          "context_recall": float,
          "per_question": list[EvalResult],
        }
    """
    import math

    # ── Attempt 1: ragas >= 0.4.x (EvaluationDataset API) ──
    try:
        from ragas import EvaluationDataset, SingleTurnSample, evaluate as ragas_evaluate
        from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
        from ragas.llms import llm_factory
        from ragas.embeddings import embedding_factory
        from openai import OpenAI as _OpenAI

        _client = _OpenAI()
        llm_wrap = llm_factory("gpt-4o-mini", client=_client)
        emb_wrap = embedding_factory("openai", model="text-embedding-3-small", client=_client)

        metrics = [
            Faithfulness(llm=llm_wrap),
            AnswerRelevancy(llm=llm_wrap, embeddings=emb_wrap),
            ContextPrecision(llm=llm_wrap),
            ContextRecall(llm=llm_wrap),
        ]

        samples = [
            SingleTurnSample(
                user_input=q, response=a,
                retrieved_contexts=ctx, reference=gt,
            )
            for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
        ]
        dataset = EvaluationDataset(samples=samples)
        result = ragas_evaluate(dataset, metrics=metrics)
        df = result.to_pandas()
        print("  [OK] RAGAS 0.4.x evaluation complete")
        return _build_result_from_df(df, questions, answers, contexts, ground_truths)

    except Exception as exc:
        print(f"  [WARN] RAGAS 0.4.x failed ({type(exc).__name__}: {exc})")

    # ── Attempt 2: ragas 0.1.x / 0.2.x (HuggingFace Dataset API) ──
    try:
        from datasets import Dataset
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import (
            faithfulness as _faith,
            answer_relevancy as _ar,
            context_precision as _cp,
            context_recall as _cr,
        )

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = ragas_evaluate(dataset, metrics=[_faith, _ar, _cp, _cr])
        df = result.to_pandas()
        print("  [OK] RAGAS 0.2.x evaluation complete")
        return _build_result_from_df(df, questions, answers, contexts, ground_truths)

    except Exception as exc:
        print(f"  [WARN] RAGAS 0.2.x failed ({type(exc).__name__}: {exc})")
        print("  -> Using heuristic metrics (no LLM required)")

    # ── Fallback: heuristic scoring ──────────────────────
    return _heuristic_evaluate(questions, answers, contexts, ground_truths)


def _build_result_from_df(df, questions, answers, contexts, ground_truths) -> dict:
    """Parse ragas DataFrame → standard result dict, patching NaN with heuristics."""
    import math
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # Compute heuristic fallback for any NaN metric
    heuristic = _heuristic_evaluate(questions, answers, contexts, ground_truths)

    per_question: list[EvalResult] = []
    for i, (_, row) in enumerate(df.iterrows()):
        def _safe(col: str, idx: int) -> float:
            v = row.get(col)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                # Patch with heuristic value for that question
                hq = heuristic["per_question"][idx] if idx < len(heuristic["per_question"]) else None
                return float(getattr(hq, col, 0.0)) if hq else 0.0
            return float(v)

        per_question.append(EvalResult(
            question=str(row.get("user_input", row.get("question", ""))),
            answer=str(row.get("response", row.get("answer", ""))),
            contexts=list(row.get("retrieved_contexts", row.get("contexts", []))),
            ground_truth=str(row.get("reference", row.get("ground_truth", ""))),
            faithfulness=_safe("faithfulness", i),
            answer_relevancy=_safe("answer_relevancy", i),
            context_precision=_safe("context_precision", i),
            context_recall=_safe("context_recall", i),
        ))

    agg = {}
    for col in metric_cols:
        vals = [getattr(r, col) for r in per_question]
        agg[col] = round(sum(vals) / len(vals), 4) if vals else 0.0

    return agg | {"per_question": per_question}


# ─── Failure Analysis (Diagnostic Tree) ─────────────────


_THRESHOLDS = {
    "faithfulness": 0.85,
    "context_recall": 0.75,
    "context_precision": 0.75,
    "answer_relevancy": 0.80,
}

_DIAGNOSES = {
    "faithfulness": (
        "LLM hallucinating",
        "Tighten system prompt, lower temperature, enforce 'answer only from context'",
    ),
    "context_recall": (
        "Missing relevant chunks",
        "Improve chunking strategy or strengthen BM25 with underthesea tokenisation",
    ),
    "context_precision": (
        "Too many irrelevant chunks retrieved",
        "Add cross-encoder reranking or apply metadata filter before retrieval",
    ),
    "answer_relevancy": (
        "Answer does not address the question",
        "Rewrite prompt template; add explicit 'answer the question directly' instruction",
    ),
}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """
    Analyse the bottom-N worst-scoring questions using the Diagnostic Tree.

    Diagnostic Tree per question:
      1. avg_score = mean(faithfulness, answer_relevancy, context_precision, context_recall)
      2. worst_metric = metric with lowest score
      3. Map worst_metric → diagnosis + suggested_fix

    Returns:
        List of dicts: {question, worst_metric, score, avg_score, diagnosis, suggested_fix}
    """
    if not eval_results:
        return []

    scored = []
    for result in eval_results:
        avg = mean([
            result.faithfulness,
            result.answer_relevancy,
            result.context_precision,
            result.context_recall,
        ])
        scored.append((avg, result))

    # Sort ascending → worst first
    scored.sort(key=lambda x: x[0])
    bottom = scored[:bottom_n]

    failures: list[dict] = []
    for avg_score, result in bottom:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]
        diagnosis, suggested_fix = _DIAGNOSES[worst_metric]

        # Build Error Tree walkthrough string
        output_ok = avg_score >= 0.70
        ctx_ok = result.context_recall >= _THRESHOLDS["context_recall"]
        query_ok = result.context_precision >= _THRESHOLDS["context_precision"]
        error_tree = (
            f"Output đúng? {'Có' if output_ok else 'Không'} → "
            f"Context đúng? {'Có' if ctx_ok else 'Không'} → "
            f"Query OK? {'Có' if query_ok else 'Không'} → "
            f"Root: {diagnosis}"
        )

        failures.append({
            "question": result.question,
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "avg_score": round(avg_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
            "error_tree": error_tree,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
