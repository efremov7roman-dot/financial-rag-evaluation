# Financial RAG V2 release candidate scope

This V2 pull-request candidate was assembled from commit `ebf6c3b` with
documentation updates and a source-only CI workflow. The independently reviewed
archive remains unchanged. `release_files.json` enumerates the proposed Git
files and their SHA-256 values (except the manifest itself).
All executable source, tests, datasets, and historical benchmark results are
byte-identical to that commit.
The [artifact policy](ARTIFACT_POLICY.md) checks every local artifact path
referenced by the public documentation.

## Validated public scope

- Hybrid dense/BM25 retrieval with reciprocal rank fusion and metadata routing.
- SEC ingestion and deterministic chunk/index construction.
- Inline XBRL lookup and `Decimal` financial calculations with source
  provenance and citation checks.
- Frozen Benchmark V2: 150 retrieval questions, 27 labeled answer cases, and
  the canonical historical results in
  `experiments/benchmark_v2/final_benchmark.json`.
- Source-only tests, lint, formatting, and strict type checking. The clean
  commit archive passed 50 tests without private data or network access.

The quantitative results are historical measurements from the committed
artifacts. The release assembly did not run a model or benchmark experiment.
The six-company corpus, section-level labels, bounded retrieval pool, and
single-host latency limit generalization. The 27 answer cases test bounded
deterministic concepts; they do not validate free-form narrative answers.

## Reproducibility contract

The included `data/eval/benchmark_v2_retrieval.jsonl` and
`data/eval/benchmark_v2_answers.jsonl` are frozen inputs. Their hashes and
counts are in `experiments/benchmark_v2/manifest.json`. The included
`experiments/benchmark_v2/corpus_manifest.json` identifies exact public SEC
filings and hashes. `financial-rag reproduce-benchmark-v2` downloads those
filings, checks source and processed hashes, then builds local indexes. It
needs a descriptive SEC User-Agent and network access. Model weights, raw SEC
HTML, processed chunks, and indexes are generated locally and are excluded
from the release package. `financial-rag verify-final-benchmark` checks the
historical result against a reconstructed index; do not interpret that as a
new benchmark run.

Install from the package root with `python -m pip install -e ".[dev]"` and
run `pytest`, `ruff check .`, `ruff format --check .`, and
`mypy --strict src tests`. The candidate passed source-only checks without
private data. A wheel built using the declared build backend, installed into
a separate target, and ran `financial-rag info`. The PR candidate also resolved
all declared dependencies into a new Python 3.12 environment, installed the
package, and ran the source-only checks. Dependencies are range-constrained in
`pyproject.toml` without a lockfile. The CI workflow repeats tests, lint,
formatting, and strict mypy on Python 3.11 without fetching SEC data or model
weights.

## Excluded work

The release omits the uncommitted V3 and V4 local generation, semantic
verification, GPT-OSS judge, pilot, review, and candidate-final work. Those
are development-only or failed experiments. V4.1 and V4.2 judge QC failed;
V4.3 has not passed a fresh independent QC gate. The 70 captured pilot
answers remain unscored, and no human gold labels are approved. No claim of
Verified RAG improvement or GPT-OSS reliability belongs in this release.

Never add `.env`, local credentials, curator records, sealed V4 candidate
final cases, raw judge responses, model weights, SEC HTML cache, dense/XBRL
indexes, Python environments, test caches, or local logs. The worktree's
`.gitignore` now explicitly protects the sealed V4 candidate-final files and
curator access record; this V2 candidate includes only the committed baseline
ignore rules because the V4 data is absent entirely.

## Publication decision

This is a **reviewable, scope-limited pull-request candidate**, not a merged
GitHub release. Review the PR diff and CI result before any merge. Keep the
local experimental work and frozen artifacts in the original workspace.
