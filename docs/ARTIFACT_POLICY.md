# V2 artifact audit

The V2 README and Markdown docs have **zero broken local Markdown
links**. The path audit below covers all local `data/`, `experiments/`, and
`docs/` references in those documents. File sizes are bytes in this package.
The included files are from committed `ebf6c3b`, except the release
documentation overlays. None contains a credential identified by the
high-confidence pattern scan; public SEC excerpts and benchmark questions are
research data, not private account data.

| Referenced path | Bytes | State and decision |
| --- | ---: | --- |
| `data/eval/benchmark_v2_answers.jsonl` | 14,719 | Included; frozen 27-case answer input. |
| `data/eval/benchmark_v2_retrieval.jsonl` | 535,770 | Included; frozen 150-question retrieval input. |
| `data/eval/dense_baseline.jsonl` | 1,078 | Included; legacy documented development input. |
| `experiments/benchmark_v2/corpus_manifest.json` | 37,471 | Included; exact SEC accessions, URLs, and hashes for reconstruction. |
| `experiments/benchmark_v2/manifest.json` | 794 | Included; frozen question hashes and counts. |
| `experiments/benchmark_v2/final_benchmark.json` | 993,031 | Included; canonical historical result, not a newly run experiment. |
| `experiments/dense-baseline.json` | 3,264 | Included; legacy documented result. |
| `experiments/retrieval-comparison.json` | 17,222 | Included; legacy documented result. |
| `experiments/phase5-benchmark.json` | 106,880 | Included; legacy documented result. |
| `experiments/phase5-benchmark.md` | 946 | Included; legacy documented summary. |
| `experiments/final_benchmark.json` | 122,294 | Included; legacy documented result. |
| `experiments/benchmark_v2/` | — | Included directory; individual files are enumerated in `release_files.json`. |
| `data/index/dense` | — | Generated locally by the documented index/reproduction commands; excluded. |
| `data/index/xbrl/facts.jsonl` | — | Generated locally from the public SEC corpus; excluded. |
| `docs/BENCHMARK.md` | 7,979 | Included; methodology and denominators. |
| `docs/architecture.md` | 6,547 | Included; architecture and release boundary. |
| `docs/operations.md` | 2,247 | Included; optional service instructions. |
| `docs/RELEASE_SCOPE.md` | 4,100 | Included; this candidate's scope and limitations. |

Setup references are also accounted for: `pyproject.toml` (1,200 bytes)
declares package and test dependencies; `.env.example` (1,116 bytes) contains
empty configuration fields and is included; `release_files.json` is the
included exact-path/hash inventory. The real `.env` is ignored and excluded.
`Dockerfile` and `docker-compose.yml` are included as optional local-service
instructions; they do not embed corpus data or model weights.

The docs also link to external public SEC filing URLs; those are not local
release artifacts. `.env` is local and ignored. Raw SEC HTML, processed
chunks, model weights, dense vectors, local indexes, caches, sealed V4
candidate-final data, pilot captures, coordinator mappings, and raw judge logs
are excluded. The exact included paths and all visible uncommitted paths left
out of this package are listed in `release_files.json`.
