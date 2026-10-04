# Operations

## Local service

Prepare `data/index/dense` and `data/index/xbrl/facts.jsonl`, then run:

```powershell
financial-rag serve --host 127.0.0.1 --port 8000
```

The first query initializes the embedding runtime and can be materially slower than later queries.
Use one process per model copy. The service serializes local inference inside that process.

## Health model

- `GET /health` confirms that the HTTP process is alive and reports artifact checks. It always
  avoids model loading.
- `GET /ready` returns 200 only when the dense manifest, documents, vectors, and XBRL facts exist;
  otherwise it returns 503.
- A ready response does not pre-warm the model. Send a representative query when deployment policy
  requires warm-up.

## Query tracing

`POST /v1/query` accepts an optional `X-Request-ID` containing 1–64 letters, digits, dots,
underscores, or hyphens. Invalid or absent IDs are replaced with a generated trace ID. The response
header, response body, and JSON logs carry the same ID. Logs also include route, answer status,
duration, citation count, and question length; question text itself is not logged.

## Container

```powershell
docker compose up --build
docker compose ps
```

Compose mounts `./data` at `/app/data`. Keep the directory writable if models may need to populate
their cache. The image runs as UID 10001 and its healthcheck calls `/health` every 30 seconds.

## Benchmark and quality gate

```powershell
financial-rag reproduce-benchmark-v2
financial-rag run-final-benchmark --retrieval-dataset data/eval/benchmark_v2_retrieval.jsonl --answer-dataset data/eval/benchmark_v2_answers.jsonl --experiments-dir experiments/benchmark_v2 --skip-indexing
financial-rag verify-final-benchmark --artifact experiments/benchmark_v2/final_benchmark.json --retrieval-dataset data/eval/benchmark_v2_retrieval.jsonl --answer-dataset data/eval/benchmark_v2_answers.jsonl
pytest
ruff check .
ruff format --check .
mypy
```

The reproduction command requires a descriptive SEC User-Agent in `.env`. It validates the exact
accessions and source hashes before rebuilding the ignored corpus and indexes. The canonical JSON
benchmark artifact remains authoritative; the Markdown comparison table is generated from it.
