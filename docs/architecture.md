# Architecture

This release candidate contains the V2 retrieval, deterministic financial
fact, and evidence-provenance paths below. Local LLM generation and the
experimental V3/V4 judge pipeline are outside its build and benchmark claims.
See [release scope](RELEASE_SCOPE.md) for inclusion and limitations.

## Foundation

The package currently has three intentionally small responsibilities:

1. `config` validates settings from environment variables and an optional `.env` file.
2. `logging` configures structured JSON events on top of the standard logging system.
3. `cli` exposes basic application information and proves the installed package is runnable.

## SEC data pipeline

PHASE 1 adds a linear, reproducible ingestion path:

```text
SEC submissions -> filing discovery -> rate-limited download -> raw cache
    -> HTML cleanup -> item sections -> deterministic chunks -> JSONL
```

The SEC client identifies itself, caps request rate, retries only transient failures, and writes
downloads atomically. Each chunk retains company, ticker, CIK, filing type/date, reporting period,
section, accession number, and source URL. Fiscal year remains null until it can be established
reliably rather than inferred from a calendar date.

Only recent filings exposed directly in a company's SEC submissions payload are discovered in this
phase. Historical submissions files can be added when the intended corpus actually requires them.

Benchmark V2 reproduction bypasses mutable "latest filing" selection. Its checked-in corpus
manifest supplies exact accession metadata and SEC URLs; the workflow verifies source hashes before
using the same parser and chunker, then checks processed hashes and the final corpus fingerprint.

## Dense retrieval baseline

PHASE 2 adds a deliberately simple exact index:

```text
processed JSONL -> passage embeddings -> L2 normalization -> vectors.npy
question -> query embedding -> cosine dot product -> stable Top-K ranking
```

`manifest.json` records the embedding model, vector dimension, document count, and a SHA-256
fingerprint of the ordered corpus. Loading fails explicitly if the stored chunks, vectors, and
manifest disagree. This prevents silently querying an index built for another model or corpus.

The evaluation runner measures binary query-level Recall@K, MRR, mean latency, p50, and p95. A
query is recalled when at least one result matches its exact chunk judgment or all configured
provenance selectors. Per-query ranks and retrieved IDs remain in the report for inspection.

## Hybrid retrieval and reranking

PHASE 3 adds two independently ranked candidate sources and explicit fusion:

```text
query -> dense top 30 --+
                         +-> RRF(k=60) -> top 5
query -> BM25 top 30 ----+
```

BM25 uses the standard Okapi formula with configurable `k1` and `b`. RRF combines ranks rather
than incomparable dense/BM25 scores and uses deterministic tie breaking. An optional FastEmbed
cross encoder can rerank the fused pool; the pool size remains configurable and defaults to 5 based
on measured quality/latency tradeoffs.

Every response records per-stage latency for dense search, BM25, fusion, and reranking. The shared
experiment runner evaluates every approach against the same corpus fingerprint, dataset
fingerprint, judgments, and K.

## Financial intelligence

PHASE 4 adds a deterministic routing and evidence layer:

```text
question -> query plan (intent + company + form + period)
    |-> narrative -> hybrid retrieval -> metadata/temporal filter -> chunk citations
    |-> fact      -> local inline-XBRL lookup ---------------------> fact citations
    `-> calculate -> XBRL operands -> Decimal calculation --------> operand citations
```

The XBRL builder extracts numeric `ix:nonFraction` elements from the immutable SEC cache, resolves
their contexts and scale, attaches filing provenance from the corpus, and writes validated JSONL.
Fact selection prefers consolidated values, recognized concepts, annual duration for year queries,
and the filing closest to the requested reporting period. The engine never substitutes retrieved
text for a missing calculation operand and returns `insufficient_evidence` instead.

Grounded natural-language synthesis and external vector databases remain outside the project scope.

## Evaluation and failure analysis

PHASE 5 runs four retrieval configurations on one immutable dataset fingerprint and evaluates the
financial answer layer separately:

```text
retrieval benchmark -> E1 dense / E2 hybrid / E3 reranked / E4 routed hybrid
answer benchmark    -> route / status / values / citations / grounding / refusal
                    -> unified JSON -> failure taxonomy -> generated Markdown tables
```

The unified artifact retains configuration, corpus and dataset fingerprints, every ranking,
per-stage latency, every answer check, and every classified failure. The summary chooses a retrieval
winner by nDCG, then MRR, then latency; it does not hide failures from losing configurations.

`financial-rag reproduce-benchmark-v2` reconstructs the excluded public corpus and indexes without
regenerating or optimizing the frozen question sets.

## Production runtime

PHASE 6 exposes the same financial engine through CLI and FastAPI:

```text
HTTP client -> FastAPI validation -> trace ID -> lazy ApplicationRuntime
                                          |-> QueryRouter
                                          |-> Hybrid retrieval + filters
                                          |-> XBRL facts + Decimal calculations
                                          `-> FinancialAnswer + citations
```

`ApplicationRuntime` validates persisted dense/XBRL inputs and loads the embedding model, indexes,
and fact store only on the first query. A load lock prevents duplicate initialization and a query
lock serializes access to local inference resources. Deployments therefore use one Uvicorn worker;
horizontal scaling should create independent containers rather than multiple model copies in one
container.

`/health` is a pure liveness check. `/ready` checks the dense manifest, documents, vectors, and XBRL
fact file without loading models. Every `/v1/query` response carries an `X-Trace-ID` and a structured
trace containing start time, end-to-end duration, route, evidence status, and citation count. The
same fields are emitted as JSON logs for correlation.

The Docker image runs as a non-root user and receives indexes and model cache through `/app/data`.
No source filing download is attempted at server startup.
