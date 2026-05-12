# Demo Script

Use this as the 5-minute recording checklist required by Lab 24.

1. RAGAS evaluation live:
   `python scripts/run_eval.py --threshold faithfulness=0.75 --threshold answer_relevancy=0.70 --threshold context_precision=0.60 --threshold context_recall=0.65`

2. LLM-as-Judge:
   Open `phase-b/pairwise_results.csv`, then run `python phase-b/kappa_analysis.py`.

3. Guardrail adversarial tests:
   Open `phase-c/adversarial_test_results.csv` and show blocked DAN, jailbreak, and PII-style cases.

4. Latency benchmark:
   Run `python phase-c/full_pipeline.py` and show P50/P95/P99 for L1, L2, and L3.
