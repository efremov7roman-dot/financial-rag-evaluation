from datetime import date

import pytest

from financial_rag.evaluation.final_audit import (
    classify_reranker_drops,
    evaluate_final_retriever,
)
from financial_rag.evaluation.models import EvaluationCase
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.models import SearchResponse, SearchResult
from tests.test_dense_retrieval import make_chunk


class FixedRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self._results = results

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        return SearchResponse(
            query=query,
            latency_ms=2,
            stage_latencies_ms={"test": 2},
            results=self._results[:top_k],
        )


def _chunks() -> tuple[DocumentChunk, ...]:
    chunks = []
    for index, character in enumerate("abcdef", start=1):
        chunk = make_chunk(character * 64, f"evidence {index}", "Item 1.")
        chunks.append(
            chunk.model_copy(
                update={
                    "metadata": chunk.metadata.model_copy(
                        update={
                            "filing_date": date(2025, 1, index),
                            "report_date": date(2024, 12, 31),
                        }
                    )
                }
            )
        )
    return tuple(chunks)


def test_final_metrics_are_derived_from_one_top_ten_ranking() -> None:
    chunks = _chunks()
    results = tuple(
        SearchResult(rank=rank, score=1 / rank, chunk=chunk)
        for rank, chunk in enumerate(chunks, start=1)
    )
    case = EvaluationCase(
        question="find final",
        relevant_chunk_ids=frozenset({chunks[-1].chunk_id}),
    )

    report = evaluate_final_retriever(
        experiment_name="test",
        configuration={"top_k": 10},
        retriever=FixedRetriever(results),
        corpus_fingerprint="a" * 64,
        corpus=chunks,
        cases=[case],
        top_k=10,
    )

    assert report.metrics.recall_at_1 == 0
    assert report.metrics.recall_at_5 == 0
    assert report.metrics.recall_at_10 == 1
    assert report.metrics.precision_at_5 == 0
    assert report.metrics.mrr == pytest.approx(1 / 6)
    assert report.stage_latencies["test"].p95_ms == 2


def test_failure_taxonomy_identifies_wrong_period_and_reranker_drop() -> None:
    chunks = _chunks()
    wrong_period = chunks[0].model_copy(
        update={
            "metadata": chunks[0].metadata.model_copy(update={"report_date": date(2023, 12, 31)})
        }
    )
    response = (SearchResult(rank=1, score=1, chunk=wrong_period),)
    case = EvaluationCase(
        question="period question",
        relevant_chunk_ids=frozenset({chunks[-1].chunk_id}),
        expected_period=date(2024, 12, 31),
    )
    missed = evaluate_final_retriever(
        experiment_name="reranked",
        configuration={"top_k": 10},
        retriever=FixedRetriever(response),
        corpus_fingerprint="a" * 64,
        corpus=chunks,
        cases=[case],
        top_k=10,
    )
    hit_result = SearchResult(rank=1, score=1, chunk=chunks[-1])
    hybrid = evaluate_final_retriever(
        experiment_name="hybrid",
        configuration={"top_k": 10},
        retriever=FixedRetriever((hit_result,)),
        corpus_fingerprint="a" * 64,
        corpus=chunks,
        cases=[case],
        top_k=10,
    )

    assert missed.failures[0].category == "WRONG_PERIOD"
    classified = classify_reranker_drops(hybrid, missed)
    assert classified.failures[0].category == "RERANKER_DROP"
