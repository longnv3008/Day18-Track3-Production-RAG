# Lab 24 — Full Evaluation & Guardrail System

## Overview
This repository contains a complete non-bonus Lab 24 implementation for evaluating and protecting a Day 18 RAG system. It builds a 50-question synthetic test set, runs RAGAS-style four-metric evaluation, performs failure clustering, adds LLM-as-judge calibration artifacts, and implements a layered guardrail stack for PII, topic scope, adversarial inputs, and output safety. The implementation is designed to run in the `day18` conda environment with deterministic fallbacks, so the lab can be reproduced even when external judge APIs are unavailable.

## Setup
```powershell
conda activate day18
pip install -r requirements.txt
python scripts/generate_lab24_artifacts.py
python scripts/run_eval.py --threshold faithfulness=0.75 --threshold answer_relevancy=0.70 --threshold context_precision=0.60 --threshold context_recall=0.65
```

## Results Summary
### Phase A (RAGAS)
- Test set: 50 questions with 25 simple, 13 reasoning, 12 multi-context questions.
- Current summary is stored in `phase-a/ragas_summary.json`.
- Total eval cost: $0.00 for deterministic local fallback; log API cost here if replacing fallback with hosted RAGAS.
- Failure clusters are documented in `phase-a/failure_analysis.md`.

### Phase B (LLM-Judge)
- Pairwise judge uses swap-and-average to mitigate position bias.
- Absolute scoring covers accuracy, relevance, conciseness, and helpfulness.
- Cohen's kappa script: `python phase-b/kappa_analysis.py`.

### Phase C (Guardrails)
- PII, topic, adversarial, output guard, and latency benchmark outputs are in `phase-c/`.
- Refuse rate and accuracy can be recomputed from `phase-c/topic_guard_results.csv`.
- L1 and L3 P95 latency are measured in `phase-c/latency_benchmark.csv` and `phase-c/output_guard_results.csv`.

### Phase D (Blueprint)
- Production blueprint: `phase-d/blueprint.md`.

## Lessons Learned
Evaluation and guardrails solve different production risks. RAGAS-style metrics reveal where retrieval or generation quality is weak, while guardrails reduce harmful or out-of-scope interactions before they reach users. The most useful pattern is to log every layer separately, because aggregate pass/fail signals do not explain whether the issue came from retrieval, generation, judging, or safety filters.

The judge pipeline also needs calibration. Swap-and-average reduces obvious position bias, but length and style bias still need measurement. For production, the deterministic fallbacks here should be replaced with hosted judge models only after the human-label agreement is acceptable.

## Demo Video
Local placeholder: record a 5-minute demo showing `scripts/run_eval.py`, pairwise results, three adversarial blocks, and latency benchmark output.
