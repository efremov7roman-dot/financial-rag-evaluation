"""Strict preparation of the frozen Benchmark V2 corpus from public SEC filings."""

import hashlib
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from financial_rag.errors import ReproductionError
from financial_rag.ingestion.models import FilingRecord, FilingType, IngestionStats
from financial_rag.ingestion.pipeline import IngestionPipeline
from financial_rag.ingestion.sec_client import SECClient


class CorpusFiling(BaseModel):
    """One immutable SEC filing required by the benchmark corpus."""

    model_config = ConfigDict(frozen=True)

    company: str
    ticker: str | None = None
    cik: str = Field(pattern=r"^\d{10}$")
    filing_type: FilingType
    filing_date: date
    report_date: date
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    primary_document: str
    source_url: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    processed_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_chunk_count: int = Field(gt=0)

    def filing_record(self) -> FilingRecord:
        """Convert frozen metadata into the ingestion pipeline's validated record."""

        return FilingRecord.model_validate(
            self.model_dump(exclude={"source_sha256", "processed_sha256", "expected_chunk_count"})
        )


class BenchmarkCorpusManifest(BaseModel):
    """Frozen source and processing contract for rebuilding Benchmark V2."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = Field(ge=1)
    benchmark_version: str
    embedding_model: str
    chunk_max_chars: int = Field(gt=0)
    chunk_overlap_chars: int = Field(ge=0)
    expected_company_count: int = Field(gt=0)
    expected_filing_count: int = Field(gt=0)
    expected_chunk_count: int = Field(gt=0)
    expected_xbrl_fact_count: int = Field(gt=0)
    expected_corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_fact_index_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    retrieval_dataset: str
    retrieval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_dataset: str
    answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    filings: tuple[CorpusFiling, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> "BenchmarkCorpusManifest":
        """Reject incomplete or internally inconsistent corpus declarations."""

        accessions = [filing.accession_number for filing in self.filings]
        if self.benchmark_version != "benchmark_v2":
            raise ValueError("corpus manifest must target benchmark_v2")
        if self.schema_version != 1:
            raise ValueError("unsupported corpus manifest schema")
        if len(accessions) != self.expected_filing_count:
            raise ValueError("filing count does not match corpus manifest")
        if len(set(accessions)) != len(accessions):
            raise ValueError("corpus manifest contains duplicate accessions")
        if len({filing.cik for filing in self.filings}) != self.expected_company_count:
            raise ValueError("company count does not match corpus manifest")
        if sum(filing.expected_chunk_count for filing in self.filings) != self.expected_chunk_count:
            raise ValueError("chunk count does not match corpus manifest")
        if self.chunk_overlap_chars >= self.chunk_max_chars:
            raise ValueError("invalid corpus manifest chunk configuration")
        return self


def load_corpus_manifest(path: Path) -> BenchmarkCorpusManifest:
    """Load and validate a checked-in benchmark corpus manifest."""

    try:
        return BenchmarkCorpusManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ReproductionError(f"failed to load corpus manifest: {path}") from exc


def verify_frozen_datasets(manifest: BenchmarkCorpusManifest, project_root: Path) -> None:
    """Verify frozen ground truth before any SEC download or index build."""

    checks = (
        (project_root / manifest.retrieval_dataset, manifest.retrieval_sha256),
        (project_root / manifest.answer_dataset, manifest.answer_sha256),
    )
    for path, expected in checks:
        if _file_sha256(path) != expected:
            raise ReproductionError(f"frozen benchmark dataset hash mismatch: {path}")


def prepare_benchmark_corpus(
    manifest: BenchmarkCorpusManifest,
    client: SECClient,
    *,
    processed_dir: Path,
) -> IngestionStats:
    """Download, verify, parse, and chunk the exact frozen filing set."""

    expected_outputs = {
        (processed_dir / filing.cik / f"{filing.accession_number.replace('-', '')}.jsonl").resolve()
        for filing in manifest.filings
    }
    existing_outputs = (
        {path.resolve() for path in processed_dir.rglob("*.jsonl")}
        if processed_dir.is_dir()
        else set()
    )
    unexpected = existing_outputs - expected_outputs
    if unexpected:
        sample = sorted(str(path) for path in unexpected)[:3]
        raise ReproductionError(
            "processed corpus contains files outside Benchmark V2; remove the ignored "
            "data/processed "
            f"directory before reproducing (examples: {sample})"
        )

    records: list[FilingRecord] = []
    for filing in manifest.filings:
        record = filing.filing_record()
        downloaded = client.download_filing(record)
        if _content_sha256(downloaded.content) != filing.source_sha256:
            raise ReproductionError(
                f"SEC source hash mismatch for {record.accession_number}; refusing changed content"
            )
        records.append(record)

    pipeline = IngestionPipeline(
        client,
        processed_dir=processed_dir,
        chunk_max_chars=manifest.chunk_max_chars,
        chunk_overlap_chars=manifest.chunk_overlap_chars,
    )
    stats = pipeline.ingest_filings(records)
    for filing in manifest.filings:
        path = processed_dir / filing.cik / f"{filing.accession_number.replace('-', '')}.jsonl"
        if _file_sha256(path) != filing.processed_sha256:
            raise ReproductionError(f"processed filing hash mismatch: {filing.accession_number}")
        line_count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line)
        if line_count != filing.expected_chunk_count:
            raise ReproductionError(f"chunk count mismatch: {filing.accession_number}")
    if stats.chunks_written != manifest.expected_chunk_count:
        raise ReproductionError("reproduced corpus chunk count does not match manifest")
    return stats


def file_sha256(path: Path) -> str:
    """Expose stable file hashing for post-build artifact checks."""

    return _file_sha256(path)


def text_sha256(path: Path) -> str:
    """Hash UTF-8 text after normalizing platform line endings."""

    try:
        return _content_sha256(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReproductionError(f"required reproducibility file is unavailable: {path}") from exc


def _content_sha256(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReproductionError(f"required reproducibility file is unavailable: {path}") from exc
