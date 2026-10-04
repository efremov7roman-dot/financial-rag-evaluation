import json
from datetime import date
from pathlib import Path

from financial_rag.ingestion.models import DownloadedFiling, FilingRecord, FilingType
from financial_rag.ingestion.pipeline import IngestionPipeline


class FakeSource:
    def __init__(self, cache_path: Path) -> None:
        self._cache_path = cache_path
        self.filing = FilingRecord(
            company="Example Corp",
            ticker="EXM",
            cik="0000320193",
            filing_type="10-Q",
            filing_date=date(2025, 5, 1),
            report_date=date(2025, 3, 31),
            accession_number="0000320193-25-000004",
            primary_document="quarterly.htm",
            source_url="https://www.sec.gov/example-quarter",
        )

    def discover_filings(
        self,
        cik: str,
        *,
        forms: tuple[FilingType, ...],
        limit: int | None = None,
    ) -> list[FilingRecord]:
        assert cik == "320193"
        assert forms == ("10-Q",)
        assert limit == 1
        return [self.filing]

    def download_filing(self, filing: FilingRecord) -> DownloadedFiling:
        return DownloadedFiling(
            filing=filing,
            content="<h2>Item 2. Management Discussion</h2><p>Revenue increased.</p>",
            cache_path=str(self._cache_path),
            from_cache=False,
        )


def test_pipeline_writes_jsonl_and_statistics(tmp_path: Path) -> None:
    pipeline = IngestionPipeline(
        FakeSource(tmp_path / "raw.htm"),
        processed_dir=tmp_path / "processed",
        chunk_max_chars=200,
        chunk_overlap_chars=20,
    )

    stats = pipeline.ingest("320193", forms=("10-Q",), limit=1)

    output = tmp_path / "processed" / "0000320193" / "000032019325000004.jsonl"
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert stats.discovered_filings == 1
    assert stats.downloaded_filings == 1
    assert stats.processed_documents == 1
    assert stats.chunks_written == 1
    assert records[0]["metadata"]["filing_type"] == "10-Q"
    assert records[0]["text"] == "Revenue increased."


def test_pipeline_accepts_an_explicit_frozen_filing_set(tmp_path: Path) -> None:
    source = FakeSource(tmp_path / "raw.htm")
    pipeline = IngestionPipeline(
        source,
        processed_dir=tmp_path / "processed",
        chunk_max_chars=200,
        chunk_overlap_chars=20,
    )

    stats = pipeline.ingest_filings([source.filing])

    assert stats.discovered_filings == 1
    assert stats.processed_documents == 1
    assert stats.chunks_written == 1
