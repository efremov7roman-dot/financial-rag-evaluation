"""Rule-based finance query classification and metadata routing."""

import re
from collections.abc import Iterable

from financial_rag.ingestion.models import DocumentChunk, FilingType
from financial_rag.intelligence.models import (
    CalculationKind,
    MetadataFilter,
    QueryIntent,
    QueryPlan,
)

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")

METRIC_CONCEPTS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:SalesRevenueNet",
    ),
    "net income": ("us-gaap:NetIncomeLoss",),
    "gross profit": ("us-gaap:GrossProfit",),
    "operating income": ("us-gaap:OperatingIncomeLoss",),
    "assets": ("us-gaap:Assets",),
    "liabilities": ("us-gaap:Liabilities",),
    "operating cash flow": ("us-gaap:NetCashProvidedByUsedInOperatingActivities",),
    "research and development": ("us-gaap:ResearchAndDevelopmentExpense",),
}

_ALIASES: tuple[tuple[str, str], ...] = (
    ("gross margin", "gross margin"),
    ("gross profit", "gross profit"),
    ("net income", "net income"),
    ("net earnings", "net income"),
    ("operating income", "operating income"),
    ("operating cash flow", "operating cash flow"),
    ("cash from operations", "operating cash flow"),
    ("research and development", "research and development"),
    ("r&d", "research and development"),
    ("revenue", "revenue"),
    ("sales", "revenue"),
    ("liabilities", "liabilities"),
    ("assets", "assets"),
)


class QueryRouter:
    """Classify questions using the companies and periods present in the corpus."""

    def __init__(self, chunks: Iterable[DocumentChunk]) -> None:
        entities: dict[str, tuple[str, str | None, str]] = {}
        for chunk in chunks:
            metadata = chunk.metadata
            entities[metadata.cik] = (metadata.company, metadata.ticker, metadata.cik)
        self._entities = tuple(entities.values())

    def plan(self, query: str) -> QueryPlan:
        """Return a transparent, deterministic route for a question."""

        normalized = " ".join(query.casefold().split())
        years = tuple(dict.fromkeys(int(value) for value in _YEAR.findall(normalized)))
        company, ticker, cik = self._match_company(normalized)
        filing_types = _filing_types(normalized)
        metric = _metric(normalized)
        calculation: CalculationKind | None = None
        if "gross margin" in normalized:
            calculation = "gross_margin"
            metric = "gross margin"
        elif any(
            term in normalized for term in ("growth", "growth rate", "percent change", "% change")
        ):
            calculation = "percent_change"
        elif any(term in normalized for term in ("difference", "absolute change")):
            calculation = "difference"

        if calculation is not None and metric is not None:
            intent: QueryIntent = "calculation"
        elif metric is not None:
            intent = "structured_fact"
        else:
            intent = "narrative"

        concepts = (
            (*METRIC_CONCEPTS["gross profit"], *METRIC_CONCEPTS["revenue"])
            if metric == "gross margin"
            else METRIC_CONCEPTS.get(metric or "", ())
        )
        return QueryPlan(
            query=query,
            intent=intent,
            filters=MetadataFilter(
                company=company,
                ticker=ticker,
                cik=cik,
                filing_types=filing_types,
                report_years=years,
            ),
            years=years,
            metric=metric,
            concepts=concepts,
            calculation=calculation,
        )

    def _match_company(self, normalized: str) -> tuple[str | None, str | None, str | None]:
        for company, ticker, cik in self._entities:
            candidates = (
                *_company_aliases(company),
                cik,
                *((ticker.casefold(),) if ticker else ()),
            )
            if any(
                re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", normalized)
                for candidate in candidates
            ):
                return company, ticker, cik
        if len(self._entities) == 1:
            return self._entities[0]
        return None, None, None


def _metric(normalized: str) -> str | None:
    return next((metric for alias, metric in _ALIASES if alias in normalized), None)


def _filing_types(normalized: str) -> tuple[FilingType, ...]:
    selected: list[FilingType] = []
    if "10-k" in normalized or "annual" in normalized or "fiscal year" in normalized:
        selected.append("10-K")
    if "10-q" in normalized or "quarter" in normalized or re.search(r"\bq[1-4]\b", normalized):
        selected.append("10-Q")
    return tuple(selected)


def _company_aliases(company: str) -> tuple[str, ...]:
    normalized = re.sub(r"[^a-z0-9 ]+", " ", company.casefold())
    normalized = " ".join(normalized.split())
    short = re.sub(r"\s+(?:inc|corp|corporation|ltd|plc|llc)$", "", normalized)
    return tuple(dict.fromkeys((company.casefold(), normalized, short)))
