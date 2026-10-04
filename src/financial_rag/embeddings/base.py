"""Embedding boundary used by indexing, retrieval, and unit tests."""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class EmbeddingModel(Protocol):
    """Minimal interface for a single dense embedding model."""

    @property
    def model_name(self) -> str: ...

    def embed_passages(self, texts: list[str]) -> NDArray[np.float32]: ...

    def embed_query(self, query: str) -> NDArray[np.float32]: ...
