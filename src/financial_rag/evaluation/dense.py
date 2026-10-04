"""Deterministic Recall@K, MRR, and latency evaluation for dense retrieval."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from financial_rag.embeddings.base import EmbeddingModel
from financial_rag.errors import RetrievalError
from financial_rag.evaluation.models import (
    DenseEvaluationReport,
    EvaluationCase,
    QueryEvaluation,
)
from financial_rag.retrieval.dense_index import DenseIndex


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    """Load a non-empty JSONL dataset with line-level validation context."""

    cases: list[EvaluationCase] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalError(f"failed to read evaluation dataset {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except ValidationError as exc:
            message = f"invalid evaluation case at {path}:{line_number}"
            raise RetrievalError(message) from exc
    if not cases:
        raise RetrievalError(f"evaluation dataset is empty: {path}")
    return cases


def evaluate_dense(
    index: DenseIndex,
    embedder: EmbeddingModel,
    cases: list[EvaluationCase],
    *,
    top_k: int,
) -> DenseEvaluationReport:
    """Evaluate whether each query retrieves at least one labeled relevant chunk."""

    if not cases:
        raise RetrievalError("at least one evaluation case is required")
    query_results: list[QueryEvaluation] = []
    for case in cases:
        response = index.search(case.question, embedder, top_k=top_k)
        first_relevant_rank = next(
            (result.rank for result in response.results if case.is_relevant(result.chunk)),
            None,
        )
        recall = 1.0 if first_relevant_rank is not None else 0.0
        reciprocal_rank = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
        query_results.append(
            QueryEvaluation(
                question=case.question,
                retrieved_chunk_ids=tuple(result.chunk.chunk_id for result in response.results),
                first_relevant_rank=first_relevant_rank,
                recall=recall,
                reciprocal_rank=reciprocal_rank,
                latency_ms=response.latency_ms,
            )
        )

    recalls = np.asarray([result.recall for result in query_results], dtype=np.float64)
    reciprocal_ranks = np.asarray(
        [result.reciprocal_rank for result in query_results], dtype=np.float64
    )
    latencies = np.asarray([result.latency_ms for result in query_results], dtype=np.float64)
    return DenseEvaluationReport(
        model_name=index.manifest.model_name,
        corpus_fingerprint=index.manifest.corpus_fingerprint,
        dataset_fingerprint=_dataset_fingerprint(cases),
        evaluated_at=datetime.now(UTC),
        corpus_document_count=index.manifest.document_count,
        query_count=len(query_results),
        top_k=top_k,
        recall_at_k=float(np.mean(recalls)),
        mrr=float(np.mean(reciprocal_ranks)),
        mean_latency_ms=float(np.mean(latencies)),
        p50_latency_ms=float(np.percentile(latencies, 50)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        queries=tuple(query_results),
    )


def _dataset_fingerprint(cases: list[EvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = case.model_dump(mode="json")
        payload["relevant_chunk_ids"] = sorted(case.relevant_chunk_ids)
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_evaluation_report(report: DenseEvaluationReport, path: Path) -> None:
    """Persist a report atomically as stable, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
