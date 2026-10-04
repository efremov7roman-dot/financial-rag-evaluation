from datetime import date

from financial_rag.ingestion.models import ChunkMetadata, DocumentChunk
from financial_rag.intelligence.filters import FilteredRetriever, RoutedRetriever
from financial_rag.intelligence.models import MetadataFilter
from financial_rag.intelligence.routing import QueryRouter
from financial_rag.retrieval.models import SearchResponse, SearchResult


def _chunk(identifier: str, year: int, filing_type: str = "10-K") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=identifier * 64,
        index=0,
        text=f"Evidence for {year}",
        metadata=ChunkMetadata(
            company="Example Corp",
            ticker="EXM",
            cik="0000000001",
            filing_type=filing_type,  # type: ignore[arg-type]
            filing_date=date(year + 1, 1, 31),
            report_date=date(year, 12, 31),
            section="Item 7",
            source_url="https://www.sec.gov/example",
            accession_number=f"0000000001-{str(year + 1)[-2:]}-000001",
        ),
    )


class StaticRetriever:
    def __init__(self, chunks: tuple[DocumentChunk, ...]) -> None:
        self._chunks = chunks

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        results = tuple(
            SearchResult(rank=rank, score=1 / rank, chunk=chunk)
            for rank, chunk in enumerate(self._chunks[:top_k], start=1)
        )
        return SearchResponse(query=query, latency_ms=0, results=results)


def test_router_classifies_company_period_and_calculation() -> None:
    router = QueryRouter((_chunk("a", 2023), _chunk("b", 2024)))

    plan = router.plan("What was EXM revenue growth from 2023 to 2024 in annual filings?")

    assert plan.intent == "calculation"
    assert plan.calculation == "percent_change"
    assert plan.metric == "revenue"
    assert plan.years == (2023, 2024)
    assert plan.filters.cik == "0000000001"
    assert plan.filters.filing_types == ("10-K",)


def test_router_sends_unstructured_questions_to_narrative_retrieval() -> None:
    router = QueryRouter((_chunk("a", 2024),))

    plan = router.plan("What supply-chain risks did the company discuss in 2024?")

    assert plan.intent == "narrative"
    assert plan.filters.report_years == (2024,)


def test_router_matches_company_name_without_legal_suffix() -> None:
    first = _chunk("a", 2024)
    second = _chunk("b", 2024).model_copy(
        update={
            "metadata": _chunk("b", 2024).metadata.model_copy(
                update={"company": "Other Holdings LLC", "ticker": "OTH", "cik": "0000000002"}
            )
        }
    )
    router = QueryRouter((first, second))

    plan = router.plan("What was Example revenue in 2024?")

    assert plan.filters.cik == "0000000001"
    assert plan.intent == "structured_fact"


def test_filtered_retriever_applies_temporal_metadata() -> None:
    chunks = (_chunk("a", 2023), _chunk("b", 2024), _chunk("c", 2025, "10-Q"))
    retriever = FilteredRetriever(
        StaticRetriever(chunks),
        MetadataFilter(cik="0000000001", filing_types=("10-K",), report_years=(2024,)),
    )

    response = retriever.search("evidence", top_k=2)

    assert [result.chunk.metadata.report_date.year for result in response.results] == [2024]  # type: ignore[union-attr]
    assert "filtering" in response.stage_latencies_ms


def test_routed_retriever_builds_filters_for_each_query() -> None:
    chunks = (_chunk("a", 2023), _chunk("b", 2024))
    retriever = RoutedRetriever(StaticRetriever(chunks), QueryRouter(chunks), candidate_count=2)

    response = retriever.search("What risks were reported by EXM in 2024?", top_k=2)

    assert len(response.results) == 1
    assert response.results[0].chunk.metadata.report_date == date(2024, 12, 31)
