from datetime import date

from financial_rag.ingestion.chunking import chunk_document
from financial_rag.ingestion.models import FilingRecord, ParsedDocument, ParsedSection


def _filing() -> FilingRecord:
    return FilingRecord(
        company="Example Corp",
        ticker="EXM",
        cik="0000320193",
        filing_type="10-K",
        filing_date=date(2025, 2, 15),
        report_date=date(2024, 12, 31),
        accession_number="0000320193-25-000002",
        primary_document="annual.htm",
        source_url="https://www.sec.gov/example",
    )


def test_chunks_are_bounded_deterministic_and_traceable() -> None:
    document = ParsedDocument(
        text="unused aggregate",
        sections=(ParsedSection(name="Item 1A. Risk Factors", text="risk " * 100),),
    )

    first = chunk_document(_filing(), document, max_chars=80, overlap_chars=10)
    second = chunk_document(_filing(), document, max_chars=80, overlap_chars=10)

    assert len(first) > 1
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(len(chunk.text) <= 80 for chunk in first)
    assert first[0].metadata.accession_number == "0000320193-25-000002"
    assert first[0].metadata.section == "Item 1A. Risk Factors"
    assert first[0].metadata.fiscal_year is None
