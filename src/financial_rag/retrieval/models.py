"""Validated models for dense index and search results."""

from pydantic import BaseModel, ConfigDict, Field

from financial_rag.ingestion.models import DocumentChunk


class IndexManifest(BaseModel):
    """Compatibility and reproducibility metadata for a persisted index."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    model_name: str
    dimension: int = Field(gt=0)
    document_count: int = Field(gt=0)
    corpus_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    vectors_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class IndexBuildStats(BaseModel):
    """Measured output of one index build."""

    model_config = ConfigDict(frozen=True)

    model_name: str
    document_count: int
    dimension: int
    corpus_fingerprint: str
    indexing_latency_ms: float
    index_dir: str


class SearchResult(BaseModel):
    """One cosine-ranked result."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    score: float
    chunk: DocumentChunk


class SearchResponse(BaseModel):
    """Ranked results and measured end-to-end retrieval latency."""

    model_config = ConfigDict(frozen=True)

    query: str
    latency_ms: float = Field(ge=0)
    stage_latencies_ms: dict[str, float] = Field(default_factory=dict)
    results: tuple[SearchResult, ...]
