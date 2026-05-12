# Failure Cluster Analysis

## Bottom 10 Questions
| # | Question | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---:|---:|---:|---:|---:|---|
| 1 | Doanh thu thuan nam 2024 cua cong ty ABC la bao nhieu? [21] | simple | 0.61 | 0.608 | 0.53 | 0.55 | 0.575 | C2 |
| 2 | Loi nhuan sau thue nam 2024 la bao nhieu? [7] | simple | 0.61 | 0.613 | 0.53 | 0.55 | 0.576 | C2 |
| 3 | Trong pipeline production, module nao giup giam off-topic retrieval sa | multi_context | 0.625 | 0.62 | 0.545 | 0.565 | 0.589 | C1 |
| 4 | Doanh thu thuan nam 2024 cua cong ty ABC la bao nhieu? [6] | simple | 0.625 | 0.623 | 0.545 | 0.565 | 0.590 | C2 |
| 5 | Loi nhuan sau thue nam 2024 la bao nhieu? [12] | simple | 0.64 | 0.643 | 0.56 | 0.58 | 0.606 | C2 |
| 6 | Metric nao giup phat hien retrieval lay sai context khi tra loi ve doa | multi_context | 0.64 | 0.643 | 0.56 | 0.58 | 0.606 | C1 |
| 7 | Trong pipeline production, module nao giup giam off-topic retrieval sa | multi_context | 0.655 | 0.65 | 0.575 | 0.595 | 0.619 | C1 |
| 8 | Doanh thu thuan nam 2024 cua cong ty ABC la bao nhieu? [11] | simple | 0.655 | 0.653 | 0.575 | 0.595 | 0.619 | C2 |
| 9 | Loi nhuan sau thue nam 2024 la bao nhieu? [17] | simple | 0.67 | 0.672 | 0.59 | 0.61 | 0.635 | C2 |
| 10 | Metric nao giup phat hien retrieval lay sai context khi tra loi ve doa | multi_context | 0.67 | 0.672 | 0.59 | 0.61 | 0.635 | C1 |

## Clusters Identified
### Cluster C1: Multi-hop reasoning failures
**Pattern:** Questions require combining financial facts, policy constraints, and RAG operations.
**Examples:** reasoning and multi_context questions in the bottom table.
**Root cause:** Top-k retrieval can return only one evidence family, so synthesis misses the second fact.
**Proposed fix:** Increase retrieval top_k from 3 to 5, apply reranking, and add query decomposition for multi-context questions.

### Cluster C2: Keyword mismatch / off-topic retrieval
**Pattern:** Short simple questions with Vietnamese-English mixed terms retrieve adjacent but incomplete passages.
**Examples:** questions about leak reporting time and transfer safeguards.
**Root cause:** Token overlap retrieval underweights normalized Vietnamese variants.
**Proposed fix:** Add Vietnamese segmentation, synonym expansion for legal terms, and metadata filters by document type.
