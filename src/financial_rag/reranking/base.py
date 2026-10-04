"""Reranker boundary used by the retrieval pipeline and tests."""

from typing import Protocol


class Reranker(Protocol):
    """Score query-document pairs with a cross encoder."""

    @property
    def model_name(self) -> str: ...

    def score(self, query: str, documents: list[str]) -> list[float]: ...
