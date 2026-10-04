"""Evaluation judgments and measured dense baseline results."""

from datetime import date, datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_rag.ingestion.models import DocumentChunk, FilingType
from financial_rag.intelligence.models import AnswerStatus, QueryIntent


class EvaluationCase(BaseModel):
    """One query with explicit chunk IDs or a provenance-based relevance judgment."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    relevant_chunk_ids: frozenset[str] = frozenset()
    expected_company: str | None = None
    expected_filing: FilingType | None = None
    expected_period: date | None = None
    expected_section: str | None = None
    expected_accession_number: str | None = None
    benchmark_version: str | None = None
    category: str | None = None
    expected_source: str | None = None
    expected_evidence: str | None = None

    @model_validator(mode="after")
    def require_relevance_judgment(self) -> "EvaluationCase":
        """Prevent unlabeled questions from silently lowering metrics."""

        has_selector = any(
            value is not None
            for value in (
                self.expected_company,
                self.expected_filing,
                self.expected_period,
                self.expected_section,
                self.expected_accession_number,
            )
        )
        if not self.relevant_chunk_ids and not has_selector:
            raise ValueError("evaluation case requires a relevance judgment")
        return self

    def is_relevant(self, chunk: DocumentChunk) -> bool:
        """Evaluate a result against exact IDs or all configured provenance selectors."""

        if chunk.chunk_id in self.relevant_chunk_ids:
            return True
        if self.relevant_chunk_ids:
            return False
        metadata = chunk.metadata
        checks = (
            self.expected_company is None or metadata.company == self.expected_company,
            self.expected_filing is None or metadata.filing_type == self.expected_filing,
            self.expected_period is None or metadata.report_date == self.expected_period,
            self.expected_section is None
            or metadata.section.casefold().startswith(self.expected_section.casefold()),
            self.expected_accession_number is None
            or metadata.accession_number == self.expected_accession_number,
        )
        return all(checks)


class QueryEvaluation(BaseModel):
    """Per-query metric inputs retained for inspection."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieved_chunk_ids: tuple[str, ...]
    first_relevant_rank: int | None
    recall: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)


class DenseEvaluationReport(BaseModel):
    """Aggregate baseline metrics measured over one labeled dataset."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    corpus_fingerprint: str
    dataset_fingerprint: str
    evaluated_at: datetime
    corpus_document_count: int = Field(gt=0)
    query_count: int = Field(gt=0)
    top_k: int = Field(gt=0)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    mean_latency_ms: float = Field(ge=0)
    p50_latency_ms: float = Field(ge=0)
    p95_latency_ms: float = Field(ge=0)
    queries: tuple[QueryEvaluation, ...]


class LatencySummary(BaseModel):
    """Aggregate latency distribution for a retrieval stage."""

    model_config = ConfigDict(frozen=True)

    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class RankedQueryEvaluation(BaseModel):
    """Per-query relevance metrics for a comparable retrieval experiment."""

    model_config = ConfigDict(frozen=True)

    question: str
    retrieved_chunk_ids: tuple[str, ...]
    relevance: tuple[bool, ...]
    first_relevant_rank: int | None
    recall: float = Field(ge=0, le=1)
    precision: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    ndcg: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    stage_latencies_ms: dict[str, float]


class RetrievalEvaluationReport(BaseModel):
    """Comparable aggregate metrics for one retrieval configuration."""

    model_config = ConfigDict(frozen=True)

    experiment_name: str
    configuration: dict[str, str | int | float]
    model_name: str
    corpus_fingerprint: str
    dataset_fingerprint: str
    evaluated_at: datetime
    corpus_document_count: int = Field(gt=0)
    query_count: int = Field(gt=0)
    top_k: int = Field(gt=0)
    recall_at_k: float = Field(ge=0, le=1)
    precision_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    ndcg_at_k: float = Field(ge=0, le=1)
    latency: LatencySummary
    stage_latencies: dict[str, LatencySummary]
    queries: tuple[RankedQueryEvaluation, ...]


class RetrievalComparisonReport(BaseModel):
    """Same-corpus, same-dataset comparison of retrieval configurations."""

    model_config = ConfigDict(frozen=True)

    experiments: tuple[RetrievalEvaluationReport, ...]


FailureCategory = Literal[
    "retrieval_miss",
    "ranking_failure",
    "wrong_route",
    "missing_evidence",
    "calculation_error",
    "citation_failure",
    "refusal_error",
    "runtime_error",
]


class AnswerEvaluationCase(BaseModel):
    """Expected behavior for one finance-aware answer."""

    model_config = ConfigDict(frozen=True)

    question: str = Field(min_length=1)
    expected_status: AnswerStatus
    expected_route: QueryIntent
    expected_values: dict[str, Decimal] = Field(default_factory=dict)
    absolute_tolerance: Decimal = Field(default=Decimal(0), ge=0)
    minimum_citations: int = Field(default=0, ge=0)
    expected_company: str | None = None
    expected_period_years: tuple[int, ...] = ()
    benchmark_version: str | None = None
    expected_concept: str | None = None
    expected_accession_number: str | None = None
    expected_source: str | None = None


class AnswerQueryEvaluation(BaseModel):
    """Per-question checks for the evidence-grounded answer layer."""

    model_config = ConfigDict(frozen=True)

    question: str
    passed: bool
    checks: dict[str, bool]
    actual_status: AnswerStatus | None = None
    actual_route: QueryIntent | None = None
    actual_values: dict[str, Decimal] = Field(default_factory=dict)
    citation_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)


class AnswerEvaluationReport(BaseModel):
    """Aggregate deterministic-answer and citation metrics."""

    model_config = ConfigDict(frozen=True)

    dataset_fingerprint: str
    evaluated_at: datetime
    query_count: int = Field(gt=0)
    pass_rate: float = Field(ge=0, le=1)
    status_accuracy: float = Field(ge=0, le=1)
    route_accuracy: float = Field(ge=0, le=1)
    value_accuracy: float = Field(ge=0, le=1)
    citation_accuracy: float = Field(ge=0, le=1)
    grounded_answer_rate: float = Field(ge=0, le=1)
    evidence_accuracy: float = Field(ge=0, le=1)
    refusal_accuracy: float = Field(ge=0, le=1)
    latency: LatencySummary
    queries: tuple[AnswerQueryEvaluation, ...]


class FailureRecord(BaseModel):
    """Automatically classified evaluation failure."""

    model_config = ConfigDict(frozen=True)

    component: Literal["retrieval", "answer"]
    category: FailureCategory
    experiment: str | None = None
    question: str
    details: str


class BenchmarkSummary(BaseModel):
    """Primary benchmark numbers suitable for README reporting."""

    model_config = ConfigDict(frozen=True)

    retrieval_winner: str
    retrieval_recall_at_k: float = Field(ge=0, le=1)
    retrieval_mrr: float = Field(ge=0, le=1)
    retrieval_ndcg_at_k: float = Field(ge=0, le=1)
    answer_pass_rate: float = Field(ge=0, le=1)
    failure_count: int = Field(ge=0)


class BenchmarkReport(BaseModel):
    """Unified retrieval, answer-quality, latency, and failure artifact."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime
    summary: BenchmarkSummary
    retrieval: RetrievalComparisonReport
    answers: AnswerEvaluationReport
    failures: tuple[FailureRecord, ...]
