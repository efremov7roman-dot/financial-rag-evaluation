from datetime import date
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from financial_rag.errors import IndexBuildError, RetrievalError
from financial_rag.ingestion.models import ChunkMetadata, DocumentChunk
from financial_rag.retrieval.dense_index import DenseIndex, DenseIndexBuilder


class KeywordEmbedder:
    @property
    def model_name(self) -> str:
        return "test/keyword-embedding"

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return np.asarray(self._vector(query), dtype=np.float32)

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [float("revenue" in lowered), float("risk" in lowered)]


def make_chunk(chunk_id: str, text: str, section: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        index=0,
        text=text,
        metadata=ChunkMetadata(
            company="Example Corp",
            ticker="EXM",
            cik="0000000001",
            filing_type="10-K",
            filing_date=date(2025, 1, 31),
            report_date=date(2024, 12, 31),
            section=section,
            source_url="https://www.sec.gov/example",
            accession_number="0000000001-25-000001",
        ),
    )


def build_test_index(tmp_path: Path) -> tuple[DenseIndex, Path]:
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    chunks = [
        make_chunk("a" * 64, "Revenue increased because demand improved.", "Item 7."),
        make_chunk("b" * 64, "The company faces supply chain risks.", "Item 1A."),
    ]
    (processed_dir / "chunks.jsonl").write_text(
        "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    stats = DenseIndexBuilder(KeywordEmbedder()).build(
        processed_dir=processed_dir,
        index_dir=index_dir,
    )
    assert stats.document_count == 2
    assert stats.dimension == 2
    return DenseIndex.load(index_dir), index_dir


def test_dense_index_round_trip_and_cosine_ranking(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)

    response = index.search("Why did revenue grow?", KeywordEmbedder(), top_k=2)

    assert [result.chunk.chunk_id for result in response.results] == ["a" * 64, "b" * 64]
    assert response.results[0].score == pytest.approx(1.0)
    assert response.results[0].rank == 1
    assert response.latency_ms >= 0


def test_dense_index_rejects_wrong_query_model(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)

    class WrongModel(KeywordEmbedder):
        @property
        def model_name(self) -> str:
            return "test/wrong"

    with pytest.raises(RetrievalError, match="does not match"):
        index.search("revenue", WrongModel(), top_k=1)


def test_dense_index_detects_vector_corruption(tmp_path: Path) -> None:
    _, index_dir = build_test_index(tmp_path)
    vectors_path = index_dir / "vectors.npy"
    vectors = np.load(vectors_path, allow_pickle=False)
    vectors[0, 0] = 0.5
    np.save(vectors_path, vectors, allow_pickle=False)

    with pytest.raises(IndexBuildError, match="vector fingerprint"):
        DenseIndex.load(index_dir)
