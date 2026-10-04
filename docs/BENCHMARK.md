# Benchmark V2

Generated: 2026-10-04T21:03:41.711165+00:00

## Dataset and corpus

The benchmark uses 150 retrieval questions and 27 answer questions over 59 real SEC filings (13 10-K and 46 10-Q) from 6 companies (AMAZON COM INC, Alphabet Inc., Apple Inc., MICROSOFT CORP, Meta Platforms, Inc., NVIDIA CORP). The filings span report years 2024-2026 and contain 59 processed documents, 9166 chunks, and 60950 normalized XBRL facts. Filing dates run from 2024-04-25 to 2026-08-26.

Raw filings occupy 138.15 MiB (144,862,884 bytes), processed chunks 18.54 MiB (19,441,477 bytes), for 156.69 MiB (164,304,361 bytes) of corpus data in total. The dense index occupies 31.98 MiB (33,530,054 bytes). The vector matrix contains 9166 x 384 dimensions and occupies 13.43 MiB (14,079,104 bytes).

The frozen benchmark was constructed before retrieval runs. Every retrieval question has company, filing type, reporting period, accession, section, source URL, evidence excerpt, and exact relevant chunk IDs. Numerical cases use SEC inline-XBRL source facts and deterministic Decimal calculations.

### Question categories

| Category | Questions |
| --- | ---: |
| company_specific | 19 |
| cross_period | 19 |
| exact_financial_terminology | 19 |
| filing_specific | 19 |
| narrative_semantic | 19 |
| numerical_xbrl | 18 |
| similar_information_across_filings | 18 |
| temporal | 19 |

## Methodology

Each retriever receives the identical ordered dataset and is warmed once before timing. One top-10 call supplies Recall@1/5/10, Precision@5, MRR, and nDCG@10. Recall@K is the query-level hit rate: a question scores 1 when at least one labeled chunk is in the first K results. Latencies are local CPU measurements and include query embedding; reranker latency is also recorded separately.
A deterministic 10,000-sample paired bootstrap estimates the 95% confidence interval for the Final-minus-Dense Recall@5 difference.

## Retrieval results

| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@10 | p50 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense | 0.080 | 0.280 | 0.453 | 0.059 | 0.174 | 0.057 | 87.73 ms | 107.62 ms |
| Hybrid | 0.060 | 0.233 | 0.367 | 0.048 | 0.138 | 0.047 | 131.23 ms | 737.86 ms |
| Hybrid + Reranker | 0.107 | 0.260 | 0.367 | 0.053 | 0.177 | 0.053 | 3424.34 ms | 3727.45 ms |
| Final (Routed Hybrid) | 0.180 | 0.460 | 0.633 | 0.120 | 0.303 | 0.119 | 589.93 ms | 807.85 ms |

## Measured improvement and trade-off

Final versus Dense: Recall@5 changed by +0.180 (+64.3%), MRR by +0.129 (+74.3%), and nDCG@10 by +0.062. p95 latency changed by +700.24 ms (+650.7%).
Paired bootstrap Recall@5 difference: observed +0.180, 95% CI [+0.113, +0.253] (10,000 samples).

Final routing improved retrieval quality but increased latency versus dense retrieval.

The final pipeline is routed Hybrid RRF without the cross-encoder. This is an evidence-based choice: the reranker configuration is measured separately and is not claimed as part of production when its quality/latency trade-off is inferior.

## Failure analysis

### Dense

| Failure | Count | Share of failures |
| --- | ---: | ---: |
| WRONG_COMPANY | 1 | 1.2% |
| WRONG_PERIOD | 64 | 78.0% |
| WRONG_FILING | 0 | 0.0% |
| WRONG_SECTION | 17 | 20.7% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RERANKER_DROP | 0 | 0.0% |
| INSUFFICIENT_CONTEXT | 0 | 0.0% |
| GENERATION_ERROR | 0 | 0.0% |
| NUMERIC_REASONING_ERROR | 0 | 0.0% |
| UNSUPPORTED_ANSWER | 0 | 0.0% |

### Hybrid

| Failure | Count | Share of failures |
| --- | ---: | ---: |
| WRONG_COMPANY | 3 | 3.2% |
| WRONG_PERIOD | 72 | 75.8% |
| WRONG_FILING | 0 | 0.0% |
| WRONG_SECTION | 20 | 21.1% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RERANKER_DROP | 0 | 0.0% |
| INSUFFICIENT_CONTEXT | 0 | 0.0% |
| GENERATION_ERROR | 0 | 0.0% |
| NUMERIC_REASONING_ERROR | 0 | 0.0% |
| UNSUPPORTED_ANSWER | 0 | 0.0% |

### Hybrid + Reranker

| Failure | Count | Share of failures |
| --- | ---: | ---: |
| WRONG_COMPANY | 14 | 14.7% |
| WRONG_PERIOD | 67 | 70.5% |
| WRONG_FILING | 0 | 0.0% |
| WRONG_SECTION | 14 | 14.7% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RERANKER_DROP | 0 | 0.0% |
| INSUFFICIENT_CONTEXT | 0 | 0.0% |
| GENERATION_ERROR | 0 | 0.0% |
| NUMERIC_REASONING_ERROR | 0 | 0.0% |
| UNSUPPORTED_ANSWER | 0 | 0.0% |

### Final (Routed Hybrid)

| Failure | Count | Share of failures |
| --- | ---: | ---: |
| WRONG_COMPANY | 0 | 0.0% |
| WRONG_PERIOD | 28 | 50.9% |
| WRONG_FILING | 0 | 0.0% |
| WRONG_SECTION | 27 | 49.1% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RERANKER_DROP | 0 | 0.0% |
| INSUFFICIENT_CONTEXT | 0 | 0.0% |
| GENERATION_ERROR | 0 | 0.0% |
| NUMERIC_REASONING_ERROR | 0 | 0.0% |
| UNSUPPORTED_ANSWER | 0 | 0.0% |

### Answer pipeline

| Failure | Count | Share of failures |
| --- | ---: | ---: |
| WRONG_COMPANY | 0 | 0.0% |
| WRONG_PERIOD | 0 | 0.0% |
| WRONG_FILING | 0 | 0.0% |
| WRONG_SECTION | 0 | 0.0% |
| RETRIEVAL_MISS | 0 | 0.0% |
| RERANKER_DROP | 0 | 0.0% |
| INSUFFICIENT_CONTEXT | 0 | 0.0% |
| GENERATION_ERROR | 0 | 0.0% |
| NUMERIC_REASONING_ERROR | 0 | 0.0% |
| UNSUPPORTED_ANSWER | 0 | 0.0% |

### Top three Final/answer failure causes

- WRONG_PERIOD: 28/55 failures (50.9%).
- WRONG_SECTION: 27/55 failures (49.1%).
- No third non-zero failure category was observed.

## Financial answers and evidence grounding

| Cases | Pass | Status | Route | Values | Citations | Grounded | Evidence | Refusal | p50 | p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 27 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 9.09 ms | 27.32 ms |

Deterministic labeled cases passed: 27/27.

Citation validation resolves every cited accession and URL against the local corpus, checks company and period labels, and verifies either the exact XBRL fact or the cited chunk excerpt. No LLM is used, so token usage and LLM cost are not applicable.

## System measurements

- Indexing time: 2295720.68 ms
- Ingestion time: not reported; the audit reused the immutable SEC cache and did not repeat network downloads.
- Cross-encoder reranking p50/p95: 2307.54/2598.94 ms.
- Evidence-answer end-to-end p50/p95: 9.09/27.32 ms.

## Limitations

- The corpus contains six large US public technology companies; results should not be generalized to all issuers, industries, filing languages, or filing types.
- Section-level labels can mark multiple chunks relevant and do not measure passage completeness for long answers.
- Metadata filtering is applied to a bounded candidate pool, so the correct period can still be absent before filtering.
- Latency is a local CPU measurement and varies with host load despite model warm-up.
- Narrative output returns evidence passages rather than LLM-generated synthesis.
- XBRL answer accuracy is reported only for the labeled denominator above and is not a claim of universal financial-answer accuracy.

## Benchmark V1 versus V2

| Benchmark | Filings | Questions | Dense Recall@5 | Final Recall@5 | Dense MRR | Final MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V1 | 9 | 18 | 0.444 | 0.667 | 0.278 | 0.540 |
| V2 | 59 | 150 | 0.280 | 0.460 | 0.174 | 0.303 |

On V2, the Final pipeline retained a Recall@5 advantage over Dense.

## Reproduction

```powershell
financial-rag reproduce-benchmark-v2

financial-rag run-final-benchmark --retrieval-dataset data/eval/benchmark_v2_retrieval.jsonl --answer-dataset data/eval/benchmark_v2_answers.jsonl --experiments-dir experiments/benchmark_v2 --skip-indexing
```

The first command reconstructs the exact 59-filing public SEC corpus from
`experiments/benchmark_v2/corpus_manifest.json`, validates source and processed hashes, builds the
dense/XBRL indexes, and verifies the frozen question hashes. Large generated inputs remain outside
Git. It requires `FINANCIAL_RAG_SEC_USER_AGENT` in `.env`.

The canonical machine-readable source is `experiments/benchmark_v2/final_benchmark.json`; the four pipeline artifacts retain every ranking and stage latency separately.
