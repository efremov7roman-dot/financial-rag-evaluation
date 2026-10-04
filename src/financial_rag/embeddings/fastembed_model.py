"""CPU-only FastEmbed adapter for the dense retrieval baseline."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from financial_rag.errors import EmbeddingError


class FastEmbedModel:
    """Generate local query and passage embeddings through ONNX Runtime."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._batch_size = batch_size
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            message = f"failed to initialize embedding model {model_name!r}"
            raise EmbeddingError(message) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed passages in bounded batches."""

        if not texts:
            raise EmbeddingError("cannot embed an empty passage collection")
        try:
            vectors = list(self._model.passage_embed(texts, batch_size=self._batch_size))
        except Exception as exc:
            raise EmbeddingError("passage embedding generation failed") from exc
        return self._as_matrix(vectors, expected_count=len(texts))

    def embed_query(self, query: str) -> NDArray[np.float32]:
        """Embed one non-empty retrieval query."""

        if not query.strip():
            raise EmbeddingError("query must not be empty")
        try:
            vectors = list(self._model.query_embed(query))
        except Exception as exc:
            raise EmbeddingError("query embedding generation failed") from exc
        matrix = self._as_matrix(vectors, expected_count=1)
        return np.asarray(matrix[0], dtype=np.float32)

    @staticmethod
    def _as_matrix(vectors: Sequence[NDArray[Any]], *, expected_count: int) -> NDArray[np.float32]:
        try:
            matrix: NDArray[np.float32] = np.asarray(vectors, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise EmbeddingError("embedding model returned inconsistent vectors") from exc
        if matrix.ndim != 2 or matrix.shape[0] != expected_count or matrix.shape[1] < 1:
            message = f"embedding model returned invalid shape {matrix.shape}"
            raise EmbeddingError(message)
        if not np.isfinite(matrix).all():
            raise EmbeddingError("embedding model returned non-finite values")
        return matrix
