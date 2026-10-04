"""Candidate retrieval followed by cross-encoder reranking."""

import time

from financial_rag.errors import RetrievalError
from financial_rag.reranking.base import Reranker
from financial_rag.retrieval.base import Retriever
from financial_rag.retrieval.models import SearchResponse, SearchResult


class RerankingRetriever:
    """Rerank a fixed hybrid candidate pool and return the final cutoff."""

    def __init__(
        self,
        candidate_retriever: Retriever,
        reranker: Reranker,
        *,
        candidate_count: int = 30,
    ) -> None:
        if candidate_count < 1:
            raise ValueError("candidate_count must be positive")
        self._candidate_retriever = candidate_retriever
        self._reranker = reranker
        self._candidate_count = candidate_count

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        if top_k < 1:
            raise RetrievalError("top_k must be positive")
        started_at = time.perf_counter()
        candidates = self._candidate_retriever.search(query, top_k=self._candidate_count)
        reranking_started_at = time.perf_counter()
        scores = self._reranker.score(
            query,
            [result.chunk.text for result in candidates.results],
        )
        scored = list(zip(candidates.results, scores, strict=True))
        scored.sort(key=lambda item: (-item[1], item[0].rank, item[0].chunk.chunk_id))
        results = tuple(
            SearchResult(rank=rank, score=score, chunk=candidate.chunk)
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        )
        reranking_latency_ms = (time.perf_counter() - reranking_started_at) * 1_000
        total_latency_ms = (time.perf_counter() - started_at) * 1_000
        stage_latencies = dict(candidates.stage_latencies_ms)
        stage_latencies["reranking"] = reranking_latency_ms
        return SearchResponse(
            query=query,
            latency_ms=total_latency_ms,
            stage_latencies_ms=stage_latencies,
            results=results,
        )
