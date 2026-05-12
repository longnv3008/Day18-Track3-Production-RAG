from __future__ import annotations

import csv
import statistics
from pathlib import Path

from input_guard import InputGuard, TopicGuard
from output_guard import OutputGuard


ROOT = Path(__file__).resolve().parents[1]


def guarded_pipeline(user_input: str) -> tuple[str, dict[str, float]]:
    input_guard = InputGuard()
    topic_guard = TopicGuard()
    output_guard = OutputGuard()

    sanitized, l1_ms = input_guard.sanitize(user_input)
    topic_ok, topic_reason = topic_guard.check(sanitized)
    if not topic_ok:
        return topic_reason, {"L1": l1_ms, "L2": 0.0, "L3": 0.0}

    answer = f"Grounded RAG answer for: {sanitized[:120]}"
    safe, label, l3_ms = output_guard.check(sanitized, answer)
    if not safe:
        return f"Response blocked by output guard: {label}", {"L1": l1_ms, "L2": 0.1, "L3": l3_ms}
    return answer, {"L1": l1_ms, "L2": 0.1, "L3": l3_ms}


def benchmark() -> None:
    rows = list(csv.DictReader((ROOT / "phase-a" / "testset_v1.csv").open(encoding="utf-8")))
    timings = []
    for i in range(100):
        _, t = guarded_pipeline(rows[i % len(rows)]["question"])
        timings.append(t)
    for layer in ["L1", "L2", "L3"]:
        vals = sorted(t[layer] for t in timings)
        print(f"{layer}: P50={statistics.median(vals):.3f}ms P95={vals[94]:.3f}ms P99={vals[98]:.3f}ms")


if __name__ == "__main__":
    benchmark()
