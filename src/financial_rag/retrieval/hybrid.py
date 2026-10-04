"""Reciprocal Rank Fusion over dense and BM25 rankings."""

import time
from collections import defaultdict

from financial_rag.errors import RetrievalError
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.base import Retriever
from financial_rag.retrieval.models import SearchResponse, SearchResult


class HybridRetriever:
    """Retrieve independent dense/lexical candidates and fuse their ranks."""

    def __init__(
        self,
        dense: Retriever,
        bm25: Retriever,
        *,
        dense_candidates: int = 30,
        bm25_candidates: int = 30,
        rrf_k: int = 60,
    ) -> None:
        if min(dense_candidates, bm25_candidates, rrf_k) < 1:
            raise ValueError("candidate sizes and rrf_k must be positive")
        self._dense = dense
        self._bm25 = bm25
        self._dense_candidates = dense_candidates
        self._bm25_candidates = bm25_candidates
        self._rrf_k = rrf_k

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        if top_k < 1:
            raise RetrievalError("top_k must be positive")
        started_at = time.perf_counter()
        dense_response = self._dense.search(query, top_k=self._dense_candidates)
        bm25_response = self._bm25.search(query, top_k=self._bm25_candidates)
        fusion_started_at = time.perf_counter()
        results = rrf_fusion(
            (dense_response.results, bm25_response.results),
            rrf_k=self._rrf_k,
            top_k=top_k,
        )
        fusion_latency_ms = (time.perf_counter() - fusion_started_at) * 1_000
        total_latency_ms = (time.perf_counter() - started_at) * 1_000
        return SearchResponse(
            query=query,
            latency_ms=total_latency_ms,
            stage_latencies_ms={
                "dense": dense_response.latency_ms,
                "bm25": bm25_response.latency_ms,
                "fusion": fusion_latency_ms,
            },
            results=results,
        )


def rrf_fusion(
    rankings: tuple[tuple[SearchResult, ...], ...],
    *,
    rrf_k: int,
    top_k: int,
) -> tuple[SearchResult, ...]:
    """Fuse rankings by IDs with deterministic score/rank/ID tie breaking."""

    if rrf_k < 1 or top_k < 1:
        raise ValueError("rrf_k and top_k must be positive")
    scores: dict[str, float] = defaultdict(float)
    best_ranks: dict[str, int] = {}
    chunks: dict[str, DocumentChunk] = {}
    for ranking in rankings:
        for result in ranking:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] += 1.0 / (rrf_k + result.rank)
            best_ranks[chunk_id] = min(best_ranks.get(chunk_id, result.rank), result.rank)
            chunks[chunk_id] = result.chunk
    ordered_ids = sorted(scores, key=lambda item: (-scores[item], best_ranks[item], item))[:top_k]
    return tuple(
        SearchResult(rank=rank, score=scores[chunk_id], chunk=chunks[chunk_id])
        for rank, chunk_id in enumerate(ordered_ids, start=1)
    )
