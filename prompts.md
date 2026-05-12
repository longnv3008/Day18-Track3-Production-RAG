# Prompts Used

## Pairwise Judge
Compare two answers for factual accuracy, relevance, and conciseness. Output JSON with winner and reason.

## Absolute Judge
Score accuracy, relevance, conciseness, and helpfulness from 1-5. Output JSON with dimension scores and overall average.

## Topic Guard
Classify whether the query is about RAG, RAGAS, evaluation, data protection, or guardrails. Return a graceful refusal for out-of-scope requests.
