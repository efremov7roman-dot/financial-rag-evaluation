# PHASE 5 benchmark

Generated: 2026-09-30T01:14:16.225311+00:00

## Retrieval comparison

| Experiment | Recall@K | Precision@K | MRR | nDCG@K | Mean ms | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 Dense | 0.444 | 0.111 | 0.270 | 0.131 | 11.28 | 29.75 |
| E2 Hybrid RRF | 0.500 | 0.200 | 0.354 | 0.217 | 15.00 | 23.98 |
| E3 Hybrid RRF + Reranker | 0.500 | 0.200 | 0.421 | 0.241 | 1200.44 | 2118.70 |
| E4 Routed Hybrid RRF | 0.667 | 0.322 | 0.519 | 0.365 | 25.68 | 50.53 |

## Evidence-grounded answer evaluation

| Pass rate | Status | Route | Values | Citations | Grounded | Evidence | Refusals | Mean ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.66 |

## Failure taxonomy

| Category | Count |
| --- | ---: |
| ranking_failure | 5 |
| retrieval_miss | 34 |

Winner: **E4 Routed Hybrid RRF**. Total classified failures: **39**.
