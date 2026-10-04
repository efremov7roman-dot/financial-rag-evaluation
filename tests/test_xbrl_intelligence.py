from datetime import date
from decimal import Decimal
from pathlib import Path

from financial_rag.ingestion.models import ChunkMetadata, DocumentChunk
from financial_rag.intelligence.engine import FinancialIntelligenceEngine
from financial_rag.intelligence.models import XBRLFact
from financial_rag.intelligence.routing import QueryRouter
from financial_rag.intelligence.xbrl import InlineXBRLStore
from financial_rag.retrieval.models import SearchResponse


class EmptyRetriever:
    def search(self, query: str, *, top_k: int) -> SearchResponse:
        return SearchResponse(query=query, latency_ms=0, results=())


def _chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="a" * 64,
        index=0,
        text="Example filing evidence.",
        metadata=ChunkMetadata(
            company="Example Corp",
            ticker="EXM",
            cik="0000000001",
            filing_type="10-K",
            filing_date=date(2025, 1, 31),
            report_date=date(2024, 12, 31),
            section="Item 8",
            source_url="https://www.sec.gov/example",
            accession_number="0000000001-25-000001",
        ),
    )


def _fact(year: int, value: str) -> XBRLFact:
    return XBRLFact(
        concept="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        value=Decimal(value),
        unit="usd",
        period_start=date(year, 1, 1),
        period_end=date(year, 12, 31),
        has_dimensions=False,
        company="Example Corp",
        ticker="EXM",
        cik="0000000001",
        filing_type="10-K",
        filing_date=date(year + 1, 1, 31),
        report_date=date(year, 12, 31),
        accession_number=f"0000000001-{str(year + 1)[-2:]}-000001",
        source_url="https://www.sec.gov/example",
    )


def test_inline_xbrl_extraction_applies_scale_and_provenance(tmp_path: Path) -> None:
    filing_dir = tmp_path / "0000000001" / "000000000125000001"
    filing_dir.mkdir(parents=True)
    (filing_dir / "filing.htm").write_text(
        """
        <html><body>
          <xbrli:context id="c-1"><xbrli:period><xbrli:startDate>2024-01-01</xbrli:startDate>
          <xbrli:endDate>2024-12-31</xbrli:endDate></xbrli:period></xbrli:context>
          <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
            contextRef="c-1" unitRef="usd" scale="6">100</ix:nonFraction>
        </body></html>
        """,
        encoding="utf-8",
    )

    store = InlineXBRLStore.extract(tmp_path, (_chunk(),))

    assert len(store.facts) == 1
    assert store.facts[0].value == Decimal("100000000")
    assert store.facts[0].period_end == date(2024, 12, 31)
    assert store.facts[0].accession_number == "0000000001-25-000001"


def test_engine_calculates_growth_and_cites_both_facts() -> None:
    chunk = _chunk()
    engine = FinancialIntelligenceEngine(
        QueryRouter((chunk,)),
        EmptyRetriever(),
        InlineXBRLStore((_fact(2023, "100"), _fact(2024, "125"))),
    )

    answer = engine.answer("What was EXM revenue growth from 2023 to 2024?")

    assert answer.status == "answered"
    assert answer.values["result"] == Decimal("25.00")
    assert len(answer.citations) == 2
    assert all("xbrl=" in citation.locator for citation in answer.citations)


def test_fact_selection_prefers_filing_closest_to_requested_period() -> None:
    original = _fact(2023, "100")
    later_comparative = original.model_copy(
        update={
            "value": Decimal("101"),
            "filing_date": date(2025, 10, 31),
            "report_date": date(2025, 9, 30),
            "accession_number": "0000000001-25-000002",
        }
    )
    store = InlineXBRLStore((later_comparative, original))

    selected = store.best_fact(
        ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",),
        cik="0000000001",
        year=2023,
        filing_types=("10-K",),
    )

    assert selected == original


def test_engine_refuses_answer_when_structured_evidence_is_missing() -> None:
    chunk = _chunk()
    engine = FinancialIntelligenceEngine(
        QueryRouter((chunk,)),
        EmptyRetriever(),
        InlineXBRLStore(()),
    )

    answer = engine.answer("What was EXM revenue in 2024?")

    assert answer.status == "insufficient_evidence"
    assert answer.citations == ()
    assert answer.answer.startswith("Insufficient evidence:")
