from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "phase-a" / "ragas_summary.json"


def parse_threshold(raw: str) -> tuple[str, float]:
    name, value = raw.split("=", 1)
    return name.strip(), float(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Lab 24 eval threshold gate.")
    parser.add_argument("--threshold", action="append", default=[], help="metric=value, e.g. faithfulness=0.75")
    args = parser.parse_args()

    if not SUMMARY.exists():
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "generate_lab24_artifacts.py")])

    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    failed = []
    for metric, threshold in map(parse_threshold, args.threshold):
        score = float(summary.get(metric, 0.0))
        print(f"{metric}: {score:.3f} (threshold {threshold:.3f})")
        if score < threshold:
            failed.append((metric, score, threshold))

    if failed:
        print("Evaluation gate failed:")
        for metric, score, threshold in failed:
            print(f"- {metric}: {score:.3f} < {threshold:.3f}")
        return 1

    print("Evaluation gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
