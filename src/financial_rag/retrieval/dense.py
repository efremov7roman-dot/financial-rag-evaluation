"""Bound dense retriever implementing the shared retrieval interface."""

from financial_rag.embeddings.base import EmbeddingModel
from financial_rag.retrieval.dense_index import DenseIndex
from financial_rag.retrieval.models import SearchResponse


class DenseRetriever:
    """Bind an index and embedding model for repeated evaluation queries."""

    def __init__(self, index: DenseIndex, embedder: EmbeddingModel) -> None:
        self._index = index
        self._embedder = embedder

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        return self._index.search(query, self._embedder, top_k=top_k)
