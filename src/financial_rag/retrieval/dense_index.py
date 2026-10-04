"""Small exact cosine-similarity index for the dense baseline."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import ValidationError

from financial_rag.embeddings.base import EmbeddingModel
from financial_rag.errors import IndexBuildError, RetrievalError
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.models import (
    IndexBuildStats,
    IndexManifest,
    SearchResponse,
    SearchResult,
)

MANIFEST_FILE = "manifest.json"
DOCUMENTS_FILE = "documents.jsonl"
VECTORS_FILE = "vectors.npy"


class DenseIndexBuilder:
    """Build and persist an exact dense index from processed chunk JSONL files."""

    def __init__(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder

    def build(self, *, processed_dir: Path, index_dir: Path) -> IndexBuildStats:
        started_at = time.perf_counter()
        chunks = _load_processed_chunks(processed_dir)
        vectors = self._embedder.embed_passages([chunk.text for chunk in chunks])
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            message = "embedding count does not match processed chunk count"
            raise IndexBuildError(message)
        normalized_vectors = _normalize_matrix(vectors)
        fingerprint = _corpus_fingerprint(chunks, self._embedder.model_name)
        manifest = IndexManifest(
            model_name=self._embedder.model_name,
            dimension=normalized_vectors.shape[1],
            document_count=len(chunks),
            corpus_fingerprint=fingerprint,
            vectors_fingerprint=_vectors_fingerprint(normalized_vectors),
        )
        _persist_index(index_dir, manifest, chunks, normalized_vectors)
        latency_ms = (time.perf_counter() - started_at) * 1_000
        return IndexBuildStats(
            model_name=manifest.model_name,
            document_count=manifest.document_count,
            dimension=manifest.dimension,
            corpus_fingerprint=manifest.corpus_fingerprint,
            indexing_latency_ms=latency_ms,
            index_dir=str(index_dir),
        )


class DenseIndex:
    """Loaded exact index with stable cosine ranking."""

    def __init__(
        self,
        manifest: IndexManifest,
        chunks: list[DocumentChunk],
        vectors: NDArray[np.float32],
    ) -> None:
        if vectors.ndim != 2:
            raise IndexBuildError(f"index vectors must be a matrix, got shape {vectors.shape}")
        if vectors.shape != (manifest.document_count, manifest.dimension):
            message = "index vector shape does not match manifest"
            raise IndexBuildError(message)
        if len(chunks) != manifest.document_count:
            raise IndexBuildError("document count does not match manifest")
        if not np.isfinite(vectors).all():
            raise IndexBuildError("index contains non-finite vectors")
        if _vectors_fingerprint(vectors) != manifest.vectors_fingerprint:
            raise IndexBuildError("index vector fingerprint mismatch")
        fingerprint = _corpus_fingerprint(chunks, manifest.model_name)
        if fingerprint != manifest.corpus_fingerprint:
            raise IndexBuildError("index corpus fingerprint mismatch")
        self.manifest = manifest
        self._chunks = chunks
        self._vectors = vectors

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """Expose immutable corpus records for lexical indexing and evaluation."""

        return tuple(self._chunks)

    @classmethod
    def load(cls, index_dir: Path) -> "DenseIndex":
        """Load and validate all persisted index artifacts."""

        try:
            manifest = IndexManifest.model_validate_json(
                (index_dir / MANIFEST_FILE).read_text(encoding="utf-8")
            )
            chunks = [
                DocumentChunk.model_validate_json(line)
                for line in (index_dir / DOCUMENTS_FILE).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            vectors = np.load(index_dir / VECTORS_FILE, allow_pickle=False).astype(
                np.float32, copy=False
            )
        except (OSError, ValueError, ValidationError) as exc:
            message = f"failed to load dense index from {index_dir}"
            raise IndexBuildError(message) from exc
        return cls(manifest, chunks, vectors)

    def search(self, query: str, embedder: EmbeddingModel, *, top_k: int) -> SearchResponse:
        """Embed a query and return exact cosine-ranked chunks."""

        if top_k < 1:
            raise RetrievalError("top_k must be positive")
        if embedder.model_name != self.manifest.model_name:
            message = "query embedding model does not match the index model"
            raise RetrievalError(message)
        started_at = time.perf_counter()
        query_vector = embedder.embed_query(query)
        if query_vector.ndim != 1 or query_vector.shape[0] != self.manifest.dimension:
            message = "query embedding dimension does not match the index"
            raise RetrievalError(message)
        norm = float(np.linalg.norm(query_vector))
        if not np.isfinite(query_vector).all() or norm == 0:
            raise RetrievalError("query embedding is invalid or has zero norm")
        scores = self._vectors @ (query_vector / norm)
        result_count = min(top_k, len(self._chunks))
        ranked_indices = np.argsort(-scores, kind="stable")[:result_count]
        results = tuple(
            SearchResult(
                rank=rank,
                score=float(scores[index]),
                chunk=self._chunks[index],
            )
            for rank, index in enumerate(ranked_indices.tolist(), start=1)
        )
        latency_ms = (time.perf_counter() - started_at) * 1_000
        return SearchResponse(
            query=query,
            latency_ms=latency_ms,
            stage_latencies_ms={"dense": latency_ms},
            results=results,
        )


def _load_processed_chunks(processed_dir: Path) -> list[DocumentChunk]:
    paths = sorted(processed_dir.rglob("*.jsonl")) if processed_dir.is_dir() else []
    chunks: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    try:
        for path in paths:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                chunk = DocumentChunk.model_validate_json(line)
                if chunk.chunk_id in seen_ids:
                    raise IndexBuildError(f"duplicate chunk_id: {chunk.chunk_id}")
                seen_ids.add(chunk.chunk_id)
                chunks.append(chunk)
    except (OSError, ValidationError) as exc:
        raise IndexBuildError(f"failed to load processed chunks from {processed_dir}") from exc
    if not chunks:
        raise IndexBuildError(f"no processed chunks found under {processed_dir}")
    return chunks


def _normalize_matrix(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    if vectors.ndim != 2 or vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise IndexBuildError(f"invalid embedding matrix shape: {vectors.shape}")
    if not np.isfinite(vectors).all():
        raise IndexBuildError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IndexBuildError("embedding matrix contains a zero vector")
    return np.asarray(vectors / norms, dtype=np.float32)


def _corpus_fingerprint(chunks: list[DocumentChunk], model_name: str) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x00")
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
    return digest.hexdigest()


def _vectors_fingerprint(vectors: NDArray[np.float32]) -> str:
    canonical = np.asarray(vectors, dtype="<f4", order="C")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _persist_index(
    index_dir: Path,
    manifest: IndexManifest,
    chunks: list[DocumentChunk],
    vectors: NDArray[np.float32],
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    documents_path = index_dir / DOCUMENTS_FILE
    temporary_documents = index_dir / f"{DOCUMENTS_FILE}.tmp"
    temporary_documents.write_text(
        "".join(f"{chunk.model_dump_json()}\n" for chunk in chunks),
        encoding="utf-8",
    )
    temporary_documents.replace(documents_path)

    vectors_path = index_dir / VECTORS_FILE
    temporary_vectors = index_dir / f"{VECTORS_FILE}.tmp"
    with temporary_vectors.open("wb") as output:
        np.save(output, vectors, allow_pickle=False)
    temporary_vectors.replace(vectors_path)

    manifest_path = index_dir / MANIFEST_FILE
    temporary_manifest = index_dir / f"{MANIFEST_FILE}.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
