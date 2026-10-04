"""Common retrieval interface used by comparable experiments."""

from typing import Protocol

from financial_rag.retrieval.models import SearchResponse


class Retriever(Protocol):
    """Return a ranked list for a query at an explicit cutoff."""

    def search(self, query: str, *, top_k: int) -> SearchResponse: ...
