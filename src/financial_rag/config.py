"""Typed application configuration."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "test", "production"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Settings loaded from environment variables and an optional local ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FINANCIAL_RAG_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "financial-rag"
    environment: Environment = "development"
    log_level: LogLevel = "INFO"
    data_dir: Path = Path("data")
    sec_user_agent: str | None = None
    sec_requests_per_second: float = Field(default=5.0, gt=0, le=10)
    sec_timeout_seconds: float = Field(default=30.0, gt=0)
    sec_max_retries: int = Field(default=3, ge=0, le=8)
    chunk_max_chars: int = Field(default=2_000, ge=200)
    chunk_overlap_chars: int = Field(default=200, ge=0)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = Field(default=64, ge=1)
    retrieval_top_k: int = Field(default=5, ge=1)
    index_dir: Path = Path("data/index/dense")
    model_cache_dir: Path = Path("data/models")
    dense_candidates: int = Field(default=30, ge=1)
    bm25_candidates: int = Field(default=30, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    rerank_candidates: int = Field(default=5, ge=1)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    reranker_batch_size: int = Field(default=32, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    fact_index_path: Path = Path("data/index/xbrl/facts.jsonl")
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        """Ensure chunk overlap cannot consume an entire chunk."""

        if self.chunk_overlap_chars >= self.chunk_max_chars:
            message = "chunk_overlap_chars must be smaller than chunk_max_chars"
            raise ValueError(message)
        return self


def get_settings() -> Settings:
    """Build and validate settings from the current process environment."""

    return Settings()
