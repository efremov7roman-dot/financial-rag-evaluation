"""Orchestration for reproducible SEC filing ingestion."""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from financial_rag.ingestion.chunking import chunk_document
from financial_rag.ingestion.models import (
    DocumentChunk,
    DownloadedFiling,
    FilingRecord,
    FilingType,
    IngestionStats,
)
from financial_rag.ingestion.parser import parse_filing
from financial_rag.logging import get_logger


class FilingSource(Protocol):
    """Minimal source behavior required by the ingestion pipeline."""

    def discover_filings(
        self,
        cik: str,
        *,
        forms: tuple[FilingType, ...],
        limit: int | None = None,
    ) -> list[FilingRecord]: ...

    def download_filing(self, filing: FilingRecord) -> DownloadedFiling: ...


class IngestionPipeline:
    """Discover, cache, parse, chunk, and persist SEC filings."""

    def __init__(
        self,
        source: FilingSource,
        *,
        processed_dir: Path,
        chunk_max_chars: int,
        chunk_overlap_chars: int,
    ) -> None:
        self._source = source
        self._processed_dir = processed_dir
        self._chunk_max_chars = chunk_max_chars
        self._chunk_overlap_chars = chunk_overlap_chars
        self._logger = get_logger(component="ingestion")

    def ingest(
        self,
        cik: str,
        *,
        forms: tuple[FilingType, ...] = ("10-K", "10-Q"),
        limit: int | None = None,
    ) -> IngestionStats:
        """Run ingestion for a small, explicitly bounded filing set."""

        filings = self._source.discover_filings(cik, forms=forms, limit=limit)
        return self.ingest_filings(filings)

    def ingest_filings(self, filings: Iterable[FilingRecord]) -> IngestionStats:
        """Process an explicit filing set without changing its identity or order."""

        filing_list = list(filings)
        stats = IngestionStats(discovered_filings=len(filing_list))
        for filing in filing_list:
            downloaded = self._source.download_filing(filing)
            stats.source_bytes += len(downloaded.content.encode("utf-8"))
            if downloaded.from_cache:
                stats.cached_filings += 1
            else:
                stats.downloaded_filings += 1

            document = parse_filing(downloaded.content)
            chunks = chunk_document(
                filing,
                document,
                max_chars=self._chunk_max_chars,
                overlap_chars=self._chunk_overlap_chars,
            )
            self._write_chunks(filing, chunks)
            stats.processed_documents += 1
            stats.chunks_written += len(chunks)
            self._logger.info(
                "filing_processed",
                accession_number=filing.accession_number,
                chunks=len(chunks),
                from_cache=downloaded.from_cache,
            )

        self._logger.info("ingestion_completed", **stats.model_dump())
        return stats

    def _write_chunks(self, filing: FilingRecord, chunks: list[DocumentChunk]) -> None:
        destination = (
            self._processed_dir / filing.cik / f"{filing.accession_number.replace('-', '')}.jsonl"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = destination.with_suffix(".jsonl.tmp")
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
            for chunk in chunks:
                output.write(chunk.model_dump_json())
                output.write("\n")
        temporary_path.replace(destination)
