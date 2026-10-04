"""Thread-safe lazy application runtime shared by CLI and HTTP serving."""

import threading
import time
from pathlib import Path

from financial_rag.config import Settings
from financial_rag.embeddings import FastEmbedModel
from financial_rag.intelligence import FinancialIntelligenceEngine, InlineXBRLStore, QueryRouter
from financial_rag.intelligence.models import FinancialAnswer
from financial_rag.logging import get_logger
from financial_rag.retrieval import DenseIndex
from financial_rag.retrieval.bm25 import BM25Index
from financial_rag.retrieval.dense import DenseRetriever
from financial_rag.retrieval.hybrid import HybridRetriever


class ApplicationRuntime:
    """Load expensive models once and serialize access to the local inference stack."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._engine: FinancialIntelligenceEngine | None = None
        self._load_lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._logger = get_logger(component="runtime")

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def readiness(self) -> dict[str, bool]:
        """Check required persisted artifacts without loading models."""

        index_dir = self._settings.index_dir
        return {
            "dense_manifest": (index_dir / "manifest.json").is_file(),
            "dense_documents": (index_dir / "documents.jsonl").is_file(),
            "dense_vectors": (index_dir / "vectors.npy").is_file(),
            "xbrl_facts": self._settings.fact_index_path.is_file(),
        }

    def answer(self, question: str) -> FinancialAnswer:
        """Answer through the singleton runtime with safe local-model access."""

        engine = self._load()
        with self._query_lock:
            return engine.answer(question)

    def _load(self) -> FinancialIntelligenceEngine:
        if self._engine is not None:
            return self._engine
        with self._load_lock:
            if self._engine is not None:
                return self._engine
            started_at = time.perf_counter()
            embedder = FastEmbedModel(
                self._settings.embedding_model,
                cache_dir=self._settings.model_cache_dir,
                batch_size=self._settings.embedding_batch_size,
            )
            index = DenseIndex.load(self._settings.index_dir)
            dense = DenseRetriever(index, embedder)
            bm25 = BM25Index(index.chunks, k1=self._settings.bm25_k1, b=self._settings.bm25_b)
            hybrid = HybridRetriever(
                dense,
                bm25,
                dense_candidates=self._settings.dense_candidates,
                bm25_candidates=self._settings.bm25_candidates,
                rrf_k=self._settings.rrf_k,
            )
            fact_store = InlineXBRLStore.load(self._settings.fact_index_path)
            self._engine = FinancialIntelligenceEngine(
                QueryRouter(index.chunks),
                hybrid,
                fact_store,
                narrative_top_k=self._settings.retrieval_top_k,
                filter_candidates=(
                    self._settings.dense_candidates + self._settings.bm25_candidates
                ),
            )
            self._logger.info(
                "runtime_loaded",
                latency_ms=(time.perf_counter() - started_at) * 1_000,
                corpus_documents=index.manifest.document_count,
                corpus_fingerprint=index.manifest.corpus_fingerprint,
                fact_count=len(fact_store.facts),
            )
            return self._engine


def required_artifact_paths(settings: Settings) -> tuple[Path, ...]:
    """Expose persisted runtime inputs for deployment tooling."""

    return (
        settings.index_dir / "manifest.json",
        settings.index_dir / "documents.jsonl",
        settings.index_dir / "vectors.npy",
        settings.fact_index_path,
    )
