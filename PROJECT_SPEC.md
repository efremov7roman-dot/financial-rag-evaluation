\# Financial RAG Retrieval \& Evaluation System



\## 1. Project Goal



Build a production-style Retrieval-Augmented Generation system for financial research over real SEC filings.



This is NOT a generic "chat with PDFs" project.



The primary goal is to build, evaluate, compare, and optimize a financial information retrieval pipeline using:



\* SEC 10-K / 10-Q filings

\* SEC structured financial data / XBRL where appropriate

\* dense retrieval

\* BM25 lexical retrieval

\* hybrid retrieval

\* Reciprocal Rank Fusion (RRF)

\* cross-encoder reranking

\* metadata-aware retrieval

\* temporal filtering

\* evidence-grounded generation

\* retrieval evaluation

\* failure analysis



The project must produce measurable results suitable for a technical portfolio and resume.



The final system should be able to answer questions about public-company financial filings while returning evidence for its answers.



\---



\# 2. Core Engineering Principle



The project is evaluation-first.



Do not optimize for:



\* flashy UI

\* number of frameworks

\* number of agents

\* architectural complexity



Optimize for:



1\. retrieval quality

2\. answer grounding

3\. reproducibility

4\. measurable improvements

5\. latency

6\. maintainability

7\. explainability



Every major retrieval improvement should be measurable against a baseline.



\---



\# 3. Primary Use Cases



The system should support questions such as:



\### Narrative questions



\* What risks did the company identify in its latest 10-K?

\* Why did operating income decline?

\* What factors affected gross margin?

\* What does management say about demand?

\* What risks does management identify for the next fiscal year?



\### Temporal questions



\* How did the company's discussion of AI-related risks change between two fiscal years?

\* Compare management's explanation of revenue growth between FY2024 and FY2025.

\* What changed in the risk factors between two filings?



\### Numerical questions



\* What was revenue in a specific fiscal year?

\* How much did revenue change between two periods?

\* What was the operating margin?

\* Compare a financial metric between two reporting periods.



Narrative questions should primarily use RAG.



Structured numerical questions should use SEC structured/XBRL data when reliable structured data is available.



The LLM should not be trusted to perform critical arithmetic when deterministic calculation is possible.



\---



\# 4. Data Source



Primary source:



SEC EDGAR.



Initial filing types:



\* 10-K

\* 10-Q



8-K support is optional and should not be implemented unless explicitly requested in a later phase.



The ingestion pipeline must preserve useful metadata.



Example:



```json

{

&#x20; "company": "...",

&#x20; "ticker": "...",

&#x20; "cik": "...",

&#x20; "filing\_type": "10-K",

&#x20; "filing\_date": "...",

&#x20; "fiscal\_year": 2025,

&#x20; "section": "...",

&#x20; "source\_url": "...",

&#x20; "accession\_number": "...",

&#x20; "chunk\_id": "..."

}

```



The exact schema may evolve if justified.



\---



\# 5. Dataset Strategy



Development must begin with a small corpus.



Do NOT download a massive dataset during early development.



Recommended progression:



Stage A:



\* 3-5 companies

\* 1-2 years



Stage B:



\* 10-20 companies

\* multiple reporting periods



Stage C:



\* larger benchmark corpus only after the pipeline is stable



The final corpus size must be measured rather than predetermined.



Record:



\* number of companies

\* number of filings

\* number of documents

\* number of chunks

\* storage size

\* indexing time



These values will later be used in the final benchmark and README.



\---



\# 6. High-Level Architecture



```text

&#x20;                   SEC EDGAR

&#x20;                       |

&#x20;            +----------+----------+

&#x20;            |                     |

&#x20;            v                     v

&#x20;        Filings                  XBRL

&#x20;            |                     |

&#x20;            v                     v

&#x20;    Parsing / Cleaning      Structured Data

&#x20;            |

&#x20;            v

&#x20;         Chunking

&#x20;            |

&#x20;            v

&#x20;        Embeddings

&#x20;            |

&#x20;            v

&#x20;       Vector Index





User Question

&#x20;     |

&#x20;     v

Query Understanding

&#x20;     |

&#x20;     +---------------------------+

&#x20;     |                           |

&#x20;     v                           v

Narrative Query              Numeric Query

&#x20;     |                           |

&#x20;     v                           v

Query Processing              XBRL Lookup

&#x20;     |                           |

&#x20;     v                           v

Hybrid Retrieval            Deterministic

&#x20;     |                      Calculation

&#x20;     v                           |

Dense + BM25                     |

&#x20;     |                           |

&#x20;     v                           |

RRF Fusion                       |

&#x20;     |                           |

&#x20;     v                           |

Cross-Encoder                    |

Reranking                        |

&#x20;     |                           |

&#x20;     v                           |

Context Builder                  |

&#x20;     |                           |

&#x20;     +-------------+-------------+

&#x20;                   |

&#x20;                   v

&#x20;              Answer Builder

&#x20;                   |

&#x20;                   v

&#x20;         Answer + Evidence

```



\---



\# 7. Retrieval Pipeline



The project must implement retrieval incrementally.



\## Baseline



Dense retrieval:



```text

query

&#x20; ->

embedding

&#x20; ->

vector search

&#x20; ->

top-k chunks

```



This establishes baseline metrics.



\## Hybrid Retrieval



Combine:



\* dense semantic retrieval

\* BM25 lexical retrieval



Then fuse rankings using Reciprocal Rank Fusion.



```text

&#x20;               Query

&#x20;              /     \\

&#x20;             /       \\

&#x20;          Dense      BM25

&#x20;             \\       /

&#x20;              \\     /

&#x20;                 RRF

&#x20;                  |

&#x20;                  v

&#x20;            Candidates

```



\## Reranking



Apply a cross-encoder reranker to the retrieved candidate set.



Example:



```text

retrieve top 30

&#x20;      ->

cross-encoder

&#x20;      ->

rerank

&#x20;      ->

top 5

```



Exact K values must be configurable and determined experimentally.



Do not hardcode experimental values throughout the codebase.



\---



\# 8. Metadata-Aware Retrieval



Financial retrieval requires metadata filtering.



The system should eventually support filters such as:



\* company

\* ticker

\* filing type

\* filing date

\* fiscal year

\* section



Example query:



"Why did Microsoft's operating income change in FY2025?"



The system should avoid retrieving an otherwise semantically similar chunk from:



\* another company

\* an incorrect year

\* an irrelevant filing



Metadata must therefore be treated as a first-class retrieval signal.



\---



\# 9. Temporal Awareness



The system must correctly distinguish reporting periods.



Financial documents frequently contain similar language across multiple years.



Retrieval evaluation should therefore detect:



\* correct company / wrong year

\* correct year / wrong filing

\* correct filing / wrong section



Temporal retrieval quality should eventually be evaluated separately.



\---



\# 10. Numerical Queries



Numerical financial questions should not rely entirely on semantic RAG.



Where reliable structured SEC data exists:



```text

question

&#x20;  ->

query classification

&#x20;  ->

structured financial lookup

&#x20;  ->

deterministic calculation

&#x20;  ->

answer

```



Example:



"What was the percentage increase in revenue from FY2024 to FY2025?"



Preferred:



```text

revenue\_2024

revenue\_2025



delta = (revenue\_2025 - revenue\_2024) / revenue\_2024

```



rather than asking the LLM to infer the values and calculate them itself.



All calculations should be testable.



\---



\# 11. Evidence-Grounded Answers



Every factual answer produced from filings should contain evidence.



Where possible include:



\* company

\* filing type

\* filing date

\* fiscal period

\* section

\* source

\* retrieved passage/chunk identifier



The generation layer must receive retrieved evidence explicitly.



If sufficient evidence is not available, the system should prefer an uncertainty / insufficient-evidence response rather than fabricate an answer.



\---



\# 12. Evaluation



Evaluation is a core product feature.



It is NOT an optional final step.



The system must support an evaluation dataset containing at least:



```text

question

expected\_company

expected\_filing

expected\_period

expected\_section or evidence

expected\_answer when applicable

```



Exact schema can evolve.



\---



\# 13. Retrieval Metrics



At minimum measure:



\* Recall@K

\* Precision@K

\* MRR

\* nDCG@K



Also measure:



\* retrieval latency

\* reranking latency

\* total retrieval pipeline latency



Prefer reporting percentile latency where useful:



\* p50

\* p95



\---



\# 14. Generation Evaluation



Where feasible evaluate:



\* answer correctness

\* citation/evidence correctness

\* groundedness / faithfulness

\* answer relevance



Do not rely exclusively on LLM-as-a-judge.



Use deterministic checks where possible.



\---



\# 15. Failure Analysis



Every failed benchmark query should be classifiable.



Initial taxonomy:



\* WRONG\_COMPANY

\* WRONG\_PERIOD

\* WRONG\_FILING

\* WRONG\_SECTION

\* RETRIEVAL\_MISS

\* RERANKER\_DROP

\* INSUFFICIENT\_CONTEXT

\* GENERATION\_ERROR

\* NUMERIC\_REASONING\_ERROR

\* UNSUPPORTED\_ANSWER



Store failure-analysis results in a machine-readable format.



The purpose is to determine WHY the system fails, not only its overall score.



\---



\# 16. Experiment Tracking



Every major retrieval configuration should produce comparable results.



Minimum experiment progression:



```text

E1 Dense Retrieval

E2 Dense + BM25

E3 Hybrid + RRF

E4 Hybrid + RRF + Reranker

E5 Metadata-aware retrieval

E6 Final optimized pipeline

```



Additional experiments should only be introduced when justified by measured failure modes.



For every experiment record:



\* configuration

\* dataset version

\* corpus size

\* embedding model

\* retrieval parameters

\* reranker

\* Recall@K

\* MRR

\* nDCG

\* latency

\* timestamp/version



Results should be reproducible.



\---



\# 17. Observability



A query trace should eventually make it possible to inspect:



```text

original query

processed query

metadata filters

dense candidates

BM25 candidates

RRF scores

reranker scores

selected chunks

generation context

answer

citations

latency

token usage when applicable

cost when applicable

```



Prefer structured JSON logging.



Do not introduce a large observability platform unless justified.



\---



\# 18. Technology Guidelines



Preferred language:



Python 3.11+



Core technologies may include:



\* FastAPI

\* Pydantic

\* httpx

\* Qdrant or another justified vector store

\* sentence-transformers

\* BM25 implementation

\* cross-encoder reranker

\* pytest

\* Ruff

\* mypy

\* Docker



Use one LLM provider initially.



Do not build abstractions for multiple LLM providers unless there is a demonstrated requirement.



Frameworks such as LangChain or LlamaIndex may be used only where they clearly reduce implementation complexity.



Core retrieval logic should remain understandable and testable.



\---



\# 19. Explicit Non-Goals



Do NOT implement unless explicitly requested:



\* React

\* Next.js

\* complex frontend

\* Kubernetes

\* Kafka

\* Airflow

\* Redis

\* microservices

\* multi-agent architecture

\* knowledge graph

\* multiple cloud providers

\* multiple LLM providers

\* authentication system

\* billing system

\* elaborate admin panel



These are outside the core objective.



\---



\# 20. Project Phases



\## PHASE 0 — Foundation



Goal:



Create a clean, reproducible Python project.



Deliverables:



\* project structure

\* Python environment

\* dependency management

\* configuration

\* environment variables

\* structured logging

\* pytest

\* Ruff

\* mypy

\* README skeleton

\* basic CLI entry point

\* basic test suite



No SEC ingestion or RAG yet.



\---



\## PHASE 1 — SEC Data Pipeline



Goal:



Create a reproducible ingestion pipeline.



Deliverables:



\* SEC client

\* rate-limit-aware requests

\* filing discovery

\* 10-K / 10-Q download

\* parsing

\* text cleaning

\* section extraction where practical

\* metadata extraction

\* chunking

\* deterministic IDs

\* local caching

\* ingestion statistics

\* tests



Start with a very small number of companies.



\---



\## PHASE 2 — Dense Retrieval Baseline



Goal:



Establish the first measurable retrieval baseline.



Deliverables:



\* embedding generation

\* vector indexing

\* dense retrieval

\* configurable Top-K

\* small evaluation dataset

\* Recall@K

\* MRR

\* latency measurement

\* CLI query command



Generation may remain minimal.



The primary output of this phase is the baseline.



\---



\## PHASE 3 — Hybrid Retrieval + Reranking



Goal:



Improve retrieval quality and quantify the improvement.



Deliverables:



\* BM25

\* dense retrieval

\* RRF fusion

\* cross-encoder reranking

\* configurable candidate sizes

\* Precision@K

\* Recall@K

\* MRR

\* nDCG

\* latency comparison

\* experiment reports



Required comparison:



```text

Dense

vs

Hybrid

vs

Hybrid + Reranker

```



\---



\## PHASE 4 — Financial Intelligence



Goal:



Add finance-specific capabilities.



Deliverables:



\* query classification/routing

\* metadata filters

\* temporal filtering

\* company/period awareness

\* structured XBRL lookup

\* deterministic calculations

\* evidence citations

\* insufficient-evidence behavior

\* tests for numerical calculations



Do not add agents unless a measured requirement exists.



\---



\## PHASE 5 — Evaluation \& Failure Analysis



Goal:



Turn the system into an evaluation-driven RAG platform.



Deliverables:



\* larger benchmark

\* automated experiment runner

\* retrieval metrics

\* generation evaluation

\* latency metrics

\* failure taxonomy

\* automatic failure reports

\* machine-readable experiment artifacts

\* comparison tables



This phase should produce the primary numbers used in the README and resume.



\---



\## PHASE 6 — Production \& Portfolio



Goal:



Turn the research pipeline into a polished portfolio project.



Deliverables:



\* FastAPI

\* CLI

\* Docker

\* health endpoint

\* structured query tracing

\* final test suite

\* reproducible benchmark command

\* architecture documentation

\* benchmark tables

\* limitations

\* setup instructions

\* example queries

\* final README



Do not build a complex frontend.



\---



\# 21. Repository Structure



Use this as a guideline, not an immutable requirement:



```text

financial-rag/

|

|-- src/

|   `-- financial\_rag/

|       |-- config/

|       |-- ingestion/

|       |-- parsing/

|       |-- chunking/

|       |-- embeddings/

|       |-- retrieval/

|       |-- reranking/

|       |-- financial/

|       |-- generation/

|       |-- evaluation/

|       |-- observability/

|       `-- api/

|

|-- tests/

|

|-- data/

|   |-- raw/

|   |-- processed/

|   `-- eval/

|

|-- experiments/

|

|-- docs/

|

|-- scripts/

|

|-- .env.example

|-- .gitignore

|-- pyproject.toml

|-- Dockerfile

|-- README.md

|-- PROJECT\_SPEC.md

`-- CODEX\_MASTER\_PROMPT.md

```



Do not create empty architecture solely to match this tree.



Directories should be created when they become useful.



\---



\# 22. Testing Requirements



Core business logic must be testable without calling external APIs.



Mock external services where appropriate.



Prioritize tests for:



\* parsing

\* chunking

\* deterministic IDs

\* retrieval ranking

\* RRF

\* metrics

\* metadata filtering

\* numerical calculations

\* citation mapping



Avoid meaningless tests written only to increase coverage.



\---



\# 23. Cost Control



Development should remain inexpensive.



Rules:



1\. Use small corpora during development.

2\. Cache downloaded SEC filings.

3\. Cache embeddings where practical.

4\. Do not repeatedly embed unchanged documents.

5\. Run LLM generation only where necessary.

6\. Retrieval evaluation should work without an LLM.

7\. Prefer local/open-source embedding and reranking models when practical.

8\. Do not download the full SEC corpus.

9\. Do not introduce paid infrastructure without explicit approval.



\---



\# 24. Performance Principles



Correctness comes before optimization.



After correctness:



\* batch embeddings

\* avoid duplicate processing

\* cache immutable artifacts

\* measure before optimizing

\* profile bottlenecks

\* report p50/p95 latency



Never claim performance improvements without measurements.



\---



\# 25. Resume Integrity



Never fabricate:



\* dataset size

\* accuracy

\* Recall@K

\* MRR

\* latency

\* cost savings

\* performance improvements



All portfolio claims must be generated from reproducible experiments.



The final README should clearly distinguish:



\* measured results

\* design decisions

\* future work



\---



\# 26. Definition of Done



The project is complete when:



1\. Real SEC filings can be ingested reproducibly.

2\. Documents are parsed, chunked, embedded, and indexed.

3\. Dense retrieval has a measured baseline.

4\. Hybrid retrieval is implemented.

5\. Cross-encoder reranking is implemented.

6\. Retrieval configurations can be benchmarked against each other.

7\. Financial metadata and time periods are handled.

8\. Structured numerical queries can use XBRL/deterministic calculations.

9\. Answers include evidence.

10\. Retrieval and generation failures can be analyzed.

11\. Metrics and latency are reproducible.

12\. Tests pass.

13\. The project runs through documented commands.

14\. README contains measured benchmark results.

15\. Resume claims can be traced back to experiment artifacts.



The project should demonstrate retrieval engineering competence, not merely framework usage.



