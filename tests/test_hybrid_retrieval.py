from pathlib import Path

import pytest

from financial_rag.reranking.retriever import RerankingRetriever
from financial_rag.retrieval.bm25 import BM25Index
from financial_rag.retrieval.dense import DenseRetriever
from financial_rag.retrieval.hybrid import HybridRetriever, rrf_fusion
from financial_rag.retrieval.models import SearchResult
from tests.test_dense_retrieval import KeywordEmbedder, build_test_index


class RiskReranker:
    @property
    def model_name(self) -> str:
        return "test/risk-reranker"

    def score(self, query: str, documents: list[str]) -> list[float]:
        del query
        return [1.0 if "risk" in document.casefold() else 0.0 for document in documents]


def test_bm25_ranks_matching_financial_terms_first(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)
    bm25 = BM25Index(index.chunks)

    response = bm25.search("supply chain risk", top_k=2)

    assert response.results[0].chunk.chunk_id == "b" * 64
    assert response.results[0].score > 0
    assert response.stage_latencies_ms["bm25"] >= 0


def test_rrf_rewards_documents_present_in_both_rankings(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)
    chunks = index.chunks
    first = (
        SearchResult(rank=1, score=0.9, chunk=chunks[0]),
        SearchResult(rank=2, score=0.8, chunk=chunks[1]),
    )
    second = (
        SearchResult(rank=1, score=5.0, chunk=chunks[1]),
        SearchResult(rank=2, score=4.0, chunk=chunks[0]),
    )

    fused = rrf_fusion((first, second), rrf_k=60, top_k=2)

    assert [result.chunk.chunk_id for result in fused] == ["a" * 64, "b" * 64]
    assert fused[0].score == pytest.approx(fused[1].score)


def test_hybrid_candidates_can_be_cross_encoder_reranked(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)
    dense = DenseRetriever(index, KeywordEmbedder())
    hybrid = HybridRetriever(dense, BM25Index(index.chunks), rrf_k=60)
    reranked = RerankingRetriever(hybrid, RiskReranker(), candidate_count=2)

    response = reranked.search("revenue business question", top_k=2)

    assert response.results[0].chunk.chunk_id == "b" * 64
    assert response.stage_latencies_ms["reranking"] >= 0
    assert {"dense", "bm25", "fusion", "reranking"} <= response.stage_latencies_ms.keys()
