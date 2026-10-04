"""FastEmbed cross-encoder adapter."""

from pathlib import Path

import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder

from financial_rag.errors import RerankingError


class FastEmbedReranker:
    """Run a local CPU cross encoder over a bounded candidate list."""

    def __init__(
        self,
        model_name: str,
        *,
        cache_dir: Path,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._model_name = model_name
        self._batch_size = batch_size
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._model = TextCrossEncoder(
                model_name=model_name,
                cache_dir=str(cache_dir),
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            message = f"failed to initialize reranker model {model_name!r}"
            raise RerankingError(message) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not query.strip():
            raise RerankingError("query must not be empty")
        if not documents:
            return []
        try:
            scores = [
                float(score)
                for score in self._model.rerank(
                    query,
                    documents,
                    batch_size=self._batch_size,
                )
            ]
        except Exception as exc:
            raise RerankingError("cross-encoder scoring failed") from exc
        if len(scores) != len(documents) or not np.isfinite(scores).all():
            raise RerankingError("cross encoder returned invalid scores")
        return scores
