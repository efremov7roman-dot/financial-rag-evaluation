"""Validated models for finance-aware query planning and evidence."""

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from financial_rag.ingestion.models import FilingType

QueryIntent = Literal["narrative", "structured_fact", "calculation"]
CalculationKind = Literal["percent_change", "difference", "gross_margin"]
AnswerStatus = Literal["answered", "insufficient_evidence"]


class MetadataFilter(BaseModel):
    """Company, filing, and reporting-period constraints for retrieval."""

    model_config = ConfigDict(frozen=True)

    company: str | None = None
    ticker: str | None = None
    cik: str | None = None
    filing_types: tuple[FilingType, ...] = ()
    report_years: tuple[int, ...] = ()
    report_start: date | None = None
    report_end: date | None = None


class QueryPlan(BaseModel):
    """Deterministic routing decision derived from a financial question."""

    model_config = ConfigDict(frozen=True)

    query: str
    intent: QueryIntent
    filters: MetadataFilter
    years: tuple[int, ...] = ()
    metric: str | None = None
    concepts: tuple[str, ...] = ()
    calculation: CalculationKind | None = None


class EvidenceCitation(BaseModel):
    """Machine-readable pointer back to filing evidence."""

    model_config = ConfigDict(frozen=True)

    source_url: str
    accession_number: str
    locator: str
    excerpt: str


class XBRLFact(BaseModel):
    """Normalized numeric inline-XBRL fact with filing provenance."""

    model_config = ConfigDict(frozen=True)

    concept: str
    value: Decimal
    unit: str
    period_start: date | None = None
    period_end: date
    has_dimensions: bool
    company: str
    ticker: str | None = None
    cik: str
    filing_type: FilingType
    filing_date: date
    report_date: date | None = None
    accession_number: str
    source_url: str


class FinancialAnswer(BaseModel):
    """Evidence-grounded answer or an explicit insufficient-evidence result."""

    model_config = ConfigDict(frozen=True)

    question: str
    status: AnswerStatus
    route: QueryIntent
    answer: str
    values: dict[str, Decimal] = Field(default_factory=dict)
    citations: tuple[EvidenceCitation, ...] = ()
    plan: QueryPlan
