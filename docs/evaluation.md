# Dense baseline evaluation

`data/eval/dense_baseline.jsonl` contains four manually labeled retrieval questions for Apple
filings. Labels use filing provenance and section boundaries rather than answer strings so the
baseline evaluates retrieval only.

Sources:

- Apple 2024 Form 10-K, accession `0000320193-24-000123`, period ended 2024-09-28:
  https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm
- Apple Q3 2024 Form 10-Q, accession `0000320193-24-000081`, period ended 2024-06-29:
  https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm

The dataset is intentionally small and is a development baseline, not a representative benchmark.
Expand it only with verified relevance judgments. Do not compare experiments across different
dataset versions without recording that change.

Recall@K is query-level: a case receives 1 when any result in the first K matches its relevance
judgment and 0 otherwise. MRR uses the rank of the first matching result. The report also records
mean, p50, and p95 end-to-end query latency, including query embedding generation.

## E1 result

The first measured run indexed 9 Apple filings into 738 chunks using
`BAAI/bge-small-en-v1.5`. At K=5, the four-question dataset measured Recall@5 of 1.000 and MRR of
0.646. Mean query latency was 5.53 ms, with p50 of 5.14 ms and p95 of 6.75 ms. See
`experiments/dense-baseline.json` for fingerprints and per-query results.

## E1-E3 comparison

The PHASE 3 comparison uses the same 738 chunks, four questions, and K=5 for every configuration.

| Experiment | Recall@5 | Precision@5 | MRR | nDCG@5 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 Dense | 1.000 | 0.400 | 0.646 | 0.387 | 17.28 ms |
| E2 Hybrid RRF | 1.000 | 0.400 | 0.833 | 0.445 | 17.25 ms |
| E3 Hybrid RRF + reranker | 1.000 | 0.400 | 0.667 | 0.393 | 602.15 ms |

Hybrid RRF is the measured winner. The tested MiniLM cross encoder did not improve this benchmark;
it remains available for further experiments rather than being presented as an improvement. See
`experiments/retrieval-comparison.json` for complete stage latency and per-query rankings.
Latency is indicative only because this is a four-query CPU benchmark and is sensitive to model
warm-up and host load.

## PHASE 5 benchmark

The primary benchmark expands retrieval coverage from 4 to 18 questions across all nine Apple
filings and adds 10 evidence-grounded answer cases. Retrieval judgments identify company, form,
reporting period, accession, and section. They intentionally test temporal disambiguation between
filings with highly repetitive language.

| Experiment | Recall@5 | Precision@5 | MRR | nDCG@5 | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 Dense | 0.444 | 0.111 | 0.270 | 0.131 | 9.22 ms |
| E2 Hybrid RRF | 0.500 | 0.200 | 0.354 | 0.217 | 15.93 ms |
| E3 Hybrid RRF + reranker | 0.500 | 0.200 | 0.421 | 0.241 | 1394.66 ms |
| E4 Routed Hybrid RRF | **0.667** | **0.322** | **0.519** | **0.365** | 140.29 ms |

The answer suite checks status, route, exact/tolerant values, minimum citation counts, grounding,
refusal behavior, and latency. All 10 cases passed. This is generation-layer evaluation in the
current deterministic system: it evaluates the user-facing evidence answer, not fluency from an LLM
that the project does not yet use.

Failure analysis uses a fixed taxonomy: `retrieval_miss`, `ranking_failure`, `wrong_route`,
`missing_evidence`, `calculation_error`, `citation_failure`, `refusal_error`, and `runtime_error`.
A ranking failure means a relevant result exists at rank 4 or 5; a retrieval miss means none exists
in the first five. The canonical run found 34 misses and 5 ranking failures across all experiments.
E4 accounted for 6 misses and 2 ranking failures, making temporal candidate generation the clearest
remaining retrieval problem.

Reproduce both artifacts with:

```powershell
financial-rag run-benchmark
```

The canonical outputs are `experiments/phase5-benchmark.json` and
`experiments/phase5-benchmark.md`. The JSON artifact is authoritative; the Markdown table is
generated from it.

## Final technical audit

The later canonical audit supersedes the PHASE 5 table for portfolio claims. It evaluates one
top-10 ranking per query and reports Recall@1/5/10, Precision@5, MRR, nDCG@10, warmed p50/p95, deep
citation validation, the specification taxonomy, corpus storage, and measured indexing time.

Use `financial-rag run-final-benchmark --measure-indexing`. Its authoritative artifact is
`experiments/final_benchmark.json`; the generated human-readable analysis is `docs/BENCHMARK.md`.
