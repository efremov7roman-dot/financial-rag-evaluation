"""Validated records exchanged by the ingestion pipeline."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

FilingType = Literal["10-K", "10-Q"]


class FilingRecord(BaseModel):
    """Provenance required to locate and identify an SEC filing."""

    model_config = ConfigDict(frozen=True)

    company: str
    ticker: str | None = None
    cik: str = Field(pattern=r"^\d{10}$")
    filing_type: FilingType
    filing_date: date
    report_date: date | None = None
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    primary_document: str
    source_url: str

    @field_validator("primary_document")
    @classmethod
    def validate_primary_document(cls, value: str) -> str:
        """Reject unsafe cache paths supplied by remote metadata."""

        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            message = "primary_document must be a plain file name"
            raise ValueError(message)
        return value


class DownloadedFiling(BaseModel):
    """Downloaded filing content and cache status."""

    model_config = ConfigDict(frozen=True)

    filing: FilingRecord
    content: str
    cache_path: str
    from_cache: bool


class ParsedSection(BaseModel):
    """A best-effort section extracted from a filing."""

    model_config = ConfigDict(frozen=True)

    name: str
    text: str


class ParsedDocument(BaseModel):
    """Clean filing text with optional section boundaries."""

    model_config = ConfigDict(frozen=True)

    text: str
    sections: tuple[ParsedSection, ...]


class ChunkMetadata(BaseModel):
    """Source provenance attached to every chunk."""

    model_config = ConfigDict(frozen=True)

    company: str
    ticker: str | None
    cik: str
    filing_type: FilingType
    filing_date: date
    report_date: date | None
    fiscal_year: int | None = None
    section: str
    source_url: str
    accession_number: str


class DocumentChunk(BaseModel):
    """Deterministic section-aware text chunk."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    index: int = Field(ge=0)
    text: str
    metadata: ChunkMetadata


class IngestionStats(BaseModel):
    """Machine-readable summary of one ingestion run."""

    discovered_filings: int = 0
    downloaded_filings: int = 0
    cached_filings: int = 0
    processed_documents: int = 0
    chunks_written: int = 0
    source_bytes: int = 0
