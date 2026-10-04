# Resume metrics

All claims below are generated from `experiments/benchmark_v2/final_benchmark.json`.

- Built an evaluation-first financial retrieval system over 59 real SEC filings (13 10-K, 46 10-Q), indexing 9166 text chunks and 60950 normalized XBRL facts with source-level provenance.
- Improved Recall@5 from 0.280 to 0.460 (+0.180 absolute, +64.3% relative) and MRR from 0.174 to 0.303 on a 150-question temporal SEC benchmark using metadata-routed Dense+BM25 RRF.
- Designed deterministic XBRL calculations and evidence validation that passed 27/27 labeled answer cases with 1.000 citation-evidence accuracy; final retrieval p95 was 807.85 ms on local CPU.

Do not combine these measurements with unmeasured business-impact claims.
