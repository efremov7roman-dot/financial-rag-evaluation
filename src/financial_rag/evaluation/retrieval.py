"""Comparable ranking metrics for dense, hybrid, and reranked retrieval."""

import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from financial_rag.evaluation.models import (
    EvaluationCase,
    LatencySummary,
    RankedQueryEvaluation,
    RetrievalComparisonReport,
    RetrievalEvaluationReport,
)
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.base import Retriever


def evaluate_retriever(
    *,
    experiment_name: str,
    configuration: dict[str, str | int | float],
    model_name: str,
    corpus_fingerprint: str,
    corpus: tuple[DocumentChunk, ...],
    retriever: Retriever,
    cases: list[EvaluationCase],
    top_k: int,
) -> RetrievalEvaluationReport:
    """Evaluate one retriever against identical judgments and cutoff."""

    if not cases or top_k < 1:
        raise ValueError("evaluation requires cases and a positive top_k")
    query_metrics: list[RankedQueryEvaluation] = []
    for case in cases:
        total_relevant = sum(case.is_relevant(chunk) for chunk in corpus)
        if total_relevant == 0:
            message = f"evaluation judgment matches no indexed chunks: {case.question}"
            raise ValueError(message)
        response = retriever.search(case.question, top_k=top_k)
        relevance = tuple(case.is_relevant(result.chunk) for result in response.results)
        relevant_retrieved = sum(relevance)
        first_relevant_rank = next(
            (rank for rank, relevant in enumerate(relevance, start=1) if relevant),
            None,
        )
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        recall = 1.0 if relevant_retrieved else 0.0
        precision = relevant_retrieved / top_k
        dcg = sum(
            1.0 / math.log2(rank + 1)
            for rank, relevant in enumerate(relevance, start=1)
            if relevant
        )
        ideal_count = min(total_relevant, top_k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
        query_metrics.append(
            RankedQueryEvaluation(
                question=case.question,
                retrieved_chunk_ids=tuple(result.chunk.chunk_id for result in response.results),
                relevance=relevance,
                first_relevant_rank=first_relevant_rank,
                recall=recall,
                precision=precision,
                reciprocal_rank=reciprocal_rank,
                ndcg=ndcg,
                latency_ms=response.latency_ms,
                stage_latencies_ms=response.stage_latencies_ms,
            )
        )

    return RetrievalEvaluationReport(
        experiment_name=experiment_name,
        configuration=configuration,
        model_name=model_name,
        corpus_fingerprint=corpus_fingerprint,
        dataset_fingerprint=_dataset_fingerprint(cases),
        evaluated_at=datetime.now(UTC),
        corpus_document_count=len(corpus),
        query_count=len(cases),
        top_k=top_k,
        recall_at_k=float(np.mean([metric.recall for metric in query_metrics])),
        precision_at_k=float(np.mean([metric.precision for metric in query_metrics])),
        mrr=float(np.mean([metric.reciprocal_rank for metric in query_metrics])),
        ndcg_at_k=float(np.mean([metric.ndcg for metric in query_metrics])),
        latency=_latency_summary([metric.latency_ms for metric in query_metrics]),
        stage_latencies=_stage_latency_summaries(query_metrics),
        queries=tuple(query_metrics),
    )


def write_comparison_report(report: RetrievalComparisonReport, path: Path) -> None:
    """Persist comparison results atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _latency_summary(values: list[float]) -> LatencySummary:
    samples = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        mean_ms=float(np.mean(samples)),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
    )


def _stage_latency_summaries(
    query_metrics: list[RankedQueryEvaluation],
) -> dict[str, LatencySummary]:
    stage_names = sorted({name for metric in query_metrics for name in metric.stage_latencies_ms})
    return {
        name: _latency_summary(
            [
                metric.stage_latencies_ms[name]
                for metric in query_metrics
                if name in metric.stage_latencies_ms
            ]
        )
        for name in stage_names
    }


def _dataset_fingerprint(cases: list[EvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = case.model_dump(mode="json")
        payload["relevant_chunk_ids"] = sorted(case.relevant_chunk_ids)
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
