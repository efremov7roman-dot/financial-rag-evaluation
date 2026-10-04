"""Schemas for the defensible final technical audit and benchmark."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from financial_rag.evaluation.models import AnswerEvaluationReport, LatencySummary

FinalFailureCategory = Literal[
    "WRONG_COMPANY",
    "WRONG_PERIOD",
    "WRONG_FILING",
    "WRONG_SECTION",
    "RETRIEVAL_MISS",
    "RERANKER_DROP",
    "INSUFFICIENT_CONTEXT",
    "GENERATION_ERROR",
    "NUMERIC_REASONING_ERROR",
    "UNSUPPORTED_ANSWER",
]


class CorpusStatistics(BaseModel):
    """Programmatically measured corpus and index size."""

    model_config = ConfigDict(frozen=True)

    company_count: int = Field(ge=0)
    companies: tuple[str, ...] = ()
    filing_count: int = Field(ge=0)
    ten_k_count: int = Field(ge=0)
    ten_q_count: int = Field(ge=0)
    filing_date_min: date
    filing_date_max: date
    report_year_min: int
    report_year_max: int
    processed_document_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    xbrl_fact_count: int = Field(ge=0)
    raw_corpus_bytes: int = Field(ge=0)
    processed_corpus_bytes: int = Field(ge=0)
    total_corpus_bytes: int = Field(ge=0)
    dense_index_bytes: int = Field(ge=0)
    vector_file_bytes: int = Field(ge=0)
    vector_count: int = Field(ge=0)
    vector_dimension: int = Field(ge=0)
    retrieval_question_count: int = Field(ge=0)
    answer_question_count: int = Field(ge=0)


class FinalRetrievalMetrics(BaseModel):
    """Required multi-cutoff ranking metrics from one top-10 run."""

    model_config = ConfigDict(frozen=True)

    recall_at_1: float = Field(ge=0, le=1)
    recall_at_5: float = Field(ge=0, le=1)
    recall_at_10: float = Field(ge=0, le=1)
    precision_at_5: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_10: float = Field(ge=0, le=1)


class FinalQueryEvaluation(BaseModel):
    """Retained final ranking needed to audit every aggregate."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieved_chunk_ids: tuple[str, ...]
    relevance: tuple[bool, ...]
    first_relevant_rank: int | None
    latency_ms: float = Field(ge=0)
    stage_latencies_ms: dict[str, float]


class FinalFailure(BaseModel):
    """Objective failure classification tied to one experiment and query."""

    model_config = ConfigDict(frozen=True)

    experiment: str
    category: FinalFailureCategory
    question: str
    details: str


class FinalRetrievalExperiment(BaseModel):
    """One independently persisted final retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str
    configuration: dict[str, str | int | float | bool]
    corpus_fingerprint: str
    dataset_fingerprint: str
    evaluated_at: datetime
    query_count: int = Field(gt=0)
    retrieval_cutoff: int = Field(ge=10)
    metrics: FinalRetrievalMetrics
    latency: LatencySummary
    stage_latencies: dict[str, LatencySummary]
    queries: tuple[FinalQueryEvaluation, ...]
    failures: tuple[FinalFailure, ...]


class ImprovementMetrics(BaseModel):
    """Automatically derived final-vs-dense quality and latency deltas."""

    model_config = ConfigDict(frozen=True)

    recall_at_5_absolute: float
    recall_at_5_relative_percent: float | None
    mrr_absolute: float
    mrr_relative_percent: float | None = None
    ndcg_at_10_absolute: float
    recall_at_5_bootstrap_ci_lower: float | None = None
    recall_at_5_bootstrap_ci_upper: float | None = None
    bootstrap_samples: int | None = None
    p95_latency_absolute_ms: float
    p95_latency_relative_percent: float | None
    tradeoff: str


class SystemMeasurements(BaseModel):
    """Measured system indicators; unavailable values remain explicit nulls."""

    model_config = ConfigDict(frozen=True)

    indexing_time_ms: float | None = Field(default=None, ge=0)
    ingestion_time_ms: float | None = Field(default=None, ge=0)
    answer_latency: LatencySummary
    reranking_latency: LatencySummary
    llm_token_usage: int | None = Field(default=None, ge=0)
    llm_cost_usd: float | None = Field(default=None, ge=0)


class FinalBenchmarkReport(BaseModel):
    """Canonical final audit artifact used by README and resume claims."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    benchmark_version: str = "benchmark_v1"
    category_distribution: dict[str, int] = Field(default_factory=dict)
    methodology: dict[str, str | int | bool]
    corpus: CorpusStatistics
    dense: FinalRetrievalExperiment
    hybrid: FinalRetrievalExperiment
    hybrid_reranker: FinalRetrievalExperiment
    final_pipeline: FinalRetrievalExperiment
    improvements: ImprovementMetrics
    answer_evaluation: AnswerEvaluationReport
    answer_failures: tuple[FinalFailure, ...]
    system: SystemMeasurements
