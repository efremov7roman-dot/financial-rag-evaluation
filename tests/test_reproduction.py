import hashlib
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from financial_rag.errors import ReproductionError
from financial_rag.ingestion.models import DownloadedFiling
from financial_rag.ingestion.sec_client import SECClient
from financial_rag.reproduction import (
    BenchmarkCorpusManifest,
    CorpusFiling,
    load_corpus_manifest,
    prepare_benchmark_corpus,
    verify_frozen_datasets,
)


class ChangedSource:
    def download_filing(self, filing: object) -> DownloadedFiling:
        from financial_rag.ingestion.models import FilingRecord

        assert isinstance(filing, FilingRecord)
        return DownloadedFiling(
            filing=filing,
            content="changed SEC content",
            cache_path="ignored.htm",
            from_cache=False,
        )


def test_checked_in_corpus_manifest_matches_frozen_datasets() -> None:
    manifest = load_corpus_manifest(Path("experiments/benchmark_v2/corpus_manifest.json"))

    verify_frozen_datasets(manifest, Path.cwd())

    assert len(manifest.filings) == 59
    assert sum(filing.expected_chunk_count for filing in manifest.filings) == 9_166


def test_reproduction_rejects_changed_sec_source(tmp_path: Path) -> None:
    source = CorpusFiling(
        company="Example Corp",
        ticker="EXM",
        cik="0000000001",
        filing_type="10-Q",
        filing_date=date(2025, 5, 1),
        report_date=date(2025, 3, 31),
        accession_number="0000000001-25-000001",
        primary_document="example.htm",
        source_url="https://www.sec.gov/Archives/example.htm",
        source_sha256=hashlib.sha256(b"expected SEC content").hexdigest(),
        processed_sha256="0" * 64,
        expected_chunk_count=1,
    )
    manifest = BenchmarkCorpusManifest(
        schema_version=1,
        benchmark_version="benchmark_v2",
        embedding_model="test-model",
        chunk_max_chars=200,
        chunk_overlap_chars=20,
        expected_company_count=1,
        expected_filing_count=1,
        expected_chunk_count=1,
        expected_xbrl_fact_count=1,
        expected_corpus_fingerprint="0" * 64,
        expected_fact_index_sha256="0" * 64,
        retrieval_dataset="retrieval.jsonl",
        retrieval_sha256="0" * 64,
        answer_dataset="answers.jsonl",
        answer_sha256="0" * 64,
        filings=(source,),
    )

    with pytest.raises(ReproductionError, match="SEC source hash mismatch"):
        prepare_benchmark_corpus(
            manifest,
            cast(SECClient, ChangedSource()),
            processed_dir=tmp_path / "processed",
        )
