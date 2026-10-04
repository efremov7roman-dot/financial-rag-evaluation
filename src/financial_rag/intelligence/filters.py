"""Metadata-aware wrapper for any ranked retriever."""

import time

from financial_rag.ingestion.models import ChunkMetadata
from financial_rag.intelligence.models import MetadataFilter
from financial_rag.intelligence.routing import QueryRouter
from financial_rag.retrieval.base import Retriever
from financial_rag.retrieval.models import SearchResponse, SearchResult


class FilteredRetriever:
    """Apply company, form, and temporal constraints while preserving ranking order."""

    def __init__(self, retriever: Retriever, filters: MetadataFilter, *, candidate_count: int = 30):
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self._retriever = retriever
        self._filters = filters
        self._candidate_count = candidate_count

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        started_at = time.perf_counter()
        response = self._retriever.search(query, top_k=max(top_k, self._candidate_count))
        matches = [
            result
            for result in response.results
            if metadata_matches(result.chunk.metadata, self._filters)
        ]
        results = tuple(
            SearchResult(rank=rank, score=result.score, chunk=result.chunk)
            for rank, result in enumerate(matches[:top_k], start=1)
        )
        filtering_ms = (time.perf_counter() - started_at) * 1_000 - response.latency_ms
        total_ms = (time.perf_counter() - started_at) * 1_000
        return SearchResponse(
            query=query,
            latency_ms=total_ms,
            stage_latencies_ms={**response.stage_latencies_ms, "filtering": max(0.0, filtering_ms)},
            results=results,
        )


class RoutedRetriever:
    """Derive metadata constraints independently for every incoming query."""

    def __init__(self, retriever: Retriever, router: QueryRouter, *, candidate_count: int = 60):
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self._retriever = retriever
        self._router = router
        self._candidate_count = candidate_count

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        plan = self._router.plan(query)
        return FilteredRetriever(
            self._retriever,
            plan.filters,
            candidate_count=self._candidate_count,
        ).search(query, top_k=top_k)


def metadata_matches(metadata: ChunkMetadata, filters: MetadataFilter) -> bool:
    """Evaluate all explicitly populated metadata predicates."""

    report_date = metadata.report_date
    if filters.company and metadata.company.casefold() != filters.company.casefold():
        return False
    if filters.ticker and (metadata.ticker or "").casefold() != filters.ticker.casefold():
        return False
    if filters.cik and metadata.cik != filters.cik:
        return False
    if filters.filing_types and metadata.filing_type not in filters.filing_types:
        return False
    if filters.report_years and (
        report_date is None or report_date.year not in filters.report_years
    ):
        return False
    if filters.report_start and (report_date is None or report_date < filters.report_start):
        return False
    return not (filters.report_end and (report_date is None or report_date > filters.report_end))
