"""Multi-cutoff final benchmark, corpus audit, and defensible failure analysis."""

import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from financial_rag.evaluation.final_models import (
    CorpusStatistics,
    FinalBenchmarkReport,
    FinalFailure,
    FinalFailureCategory,
    FinalQueryEvaluation,
    FinalRetrievalExperiment,
    FinalRetrievalMetrics,
    ImprovementMetrics,
    SystemMeasurements,
)
from financial_rag.evaluation.models import (
    AnswerEvaluationCase,
    AnswerEvaluationReport,
    EvaluationCase,
    LatencySummary,
)
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.base import Retriever
from financial_rag.retrieval.models import SearchResponse


def evaluate_final_retriever(
    *,
    experiment_name: str,
    configuration: dict[str, str | int | float | bool],
    retriever: Retriever,
    corpus_fingerprint: str,
    corpus: tuple[DocumentChunk, ...],
    cases: list[EvaluationCase],
    top_k: int = 10,
) -> FinalRetrievalExperiment:
    """Measure all required cutoffs from one warmed top-10 retrieval pass."""

    if top_k < 10 or not cases:
        raise ValueError("final retrieval evaluation requires cases and top_k >= 10")
    retriever.search(cases[0].question, top_k=top_k)
    query_results: list[FinalQueryEvaluation] = []
    failures: list[FinalFailure] = []
    relevant_counts = {
        case.question: sum(case.is_relevant(chunk) for chunk in corpus) for case in cases
    }
    if any(count == 0 for count in relevant_counts.values()):
        missing = next(question for question, count in relevant_counts.items() if count == 0)
        raise ValueError(f"benchmark judgment matches no chunks: {missing}")

    for case in cases:
        response = retriever.search(case.question, top_k=top_k)
        relevance = tuple(case.is_relevant(result.chunk) for result in response.results)
        first_rank = next(
            (rank for rank, relevant in enumerate(relevance, start=1) if relevant),
            None,
        )
        query_results.append(
            FinalQueryEvaluation(
                question=case.question,
                retrieved_chunk_ids=tuple(result.chunk.chunk_id for result in response.results),
                relevance=relevance,
                first_relevant_rank=first_rank,
                latency_ms=response.latency_ms,
                stage_latencies_ms=response.stage_latencies_ms,
            )
        )
        if first_rank is None:
            failures.append(_classify_retrieval_failure(experiment_name, case, response))

    metrics = _metrics(query_results, relevant_counts, top_k=top_k)
    return FinalRetrievalExperiment(
        experiment_name=experiment_name,
        configuration=configuration,
        corpus_fingerprint=corpus_fingerprint,
        dataset_fingerprint=_dataset_fingerprint(cases),
        evaluated_at=datetime.now(UTC),
        query_count=len(cases),
        retrieval_cutoff=top_k,
        metrics=metrics,
        latency=_latency([query.latency_ms for query in query_results]),
        stage_latencies=_stage_latencies(query_results),
        queries=tuple(query_results),
        failures=tuple(failures),
    )


def classify_reranker_drops(
    hybrid: FinalRetrievalExperiment,
    reranked: FinalRetrievalExperiment,
) -> FinalRetrievalExperiment:
    """Label cases where reranking removes a hybrid hit from the top 10."""

    hybrid_hits = {
        query.question: query.first_relevant_rank is not None for query in hybrid.queries
    }
    failures = []
    for failure in reranked.failures:
        if hybrid_hits.get(failure.question, False):
            failures.append(
                FinalFailure(
                    experiment=reranked.experiment_name,
                    category="RERANKER_DROP",
                    question=failure.question,
                    details="hybrid retrieved relevant evidence in top 10 but reranking did not",
                )
            )
        else:
            failures.append(failure)
    return reranked.model_copy(update={"failures": tuple(failures)})


def measure_corpus(
    *,
    chunks: tuple[DocumentChunk, ...],
    fact_count: int,
    raw_dir: Path,
    processed_dir: Path,
    index_dir: Path,
    vector_count: int,
    vector_dimension: int,
    retrieval_questions: int,
    answer_questions: int,
) -> CorpusStatistics:
    """Compute corpus and storage numbers directly from persisted files."""

    filings = {chunk.metadata.accession_number: chunk.metadata for chunk in chunks}
    companies = {metadata.cik for metadata in filings.values()}
    company_names = sorted({metadata.company for metadata in filings.values()})
    filing_dates = [metadata.filing_date for metadata in filings.values()]
    report_years = [
        metadata.report_date.year
        for metadata in filings.values()
        if metadata.report_date is not None
    ]
    raw_bytes = _tree_bytes(raw_dir)
    processed_bytes = _tree_bytes(processed_dir)
    vector_path = index_dir / "vectors.npy"
    return CorpusStatistics(
        company_count=len(companies),
        companies=tuple(company_names),
        filing_count=len(filings),
        ten_k_count=sum(metadata.filing_type == "10-K" for metadata in filings.values()),
        ten_q_count=sum(metadata.filing_type == "10-Q" for metadata in filings.values()),
        filing_date_min=min(filing_dates),
        filing_date_max=max(filing_dates),
        report_year_min=min(report_years),
        report_year_max=max(report_years),
        processed_document_count=len(list(processed_dir.rglob("*.jsonl"))),
        chunk_count=len(chunks),
        xbrl_fact_count=fact_count,
        raw_corpus_bytes=raw_bytes,
        processed_corpus_bytes=processed_bytes,
        total_corpus_bytes=raw_bytes + processed_bytes,
        dense_index_bytes=_tree_bytes(index_dir),
        vector_file_bytes=vector_path.stat().st_size,
        vector_count=vector_count,
        vector_dimension=vector_dimension,
        retrieval_question_count=retrieval_questions,
        answer_question_count=answer_questions,
    )


def build_final_report(
    *,
    corpus: CorpusStatistics,
    dense: FinalRetrievalExperiment,
    hybrid: FinalRetrievalExperiment,
    hybrid_reranker: FinalRetrievalExperiment,
    final_pipeline: FinalRetrievalExperiment,
    answer_report: AnswerEvaluationReport,
    answer_cases: list[AnswerEvaluationCase],
    retrieval_cases: list[EvaluationCase] | None = None,
    indexing_time_ms: float | None,
) -> FinalBenchmarkReport:
    """Assemble the canonical report and compute deltas without manual arithmetic."""

    dense_metrics = dense.metrics
    final_metrics = final_pipeline.metrics
    recall_delta = final_metrics.recall_at_5 - dense_metrics.recall_at_5
    p95_delta = final_pipeline.latency.p95_ms - dense.latency.p95_ms
    bootstrap_lower, bootstrap_upper = _bootstrap_recall_delta(dense, final_pipeline)
    improvements = ImprovementMetrics(
        recall_at_5_absolute=recall_delta,
        recall_at_5_relative_percent=_relative(recall_delta, dense_metrics.recall_at_5),
        mrr_absolute=final_metrics.mrr - dense_metrics.mrr,
        mrr_relative_percent=_relative(final_metrics.mrr - dense_metrics.mrr, dense_metrics.mrr),
        ndcg_at_10_absolute=final_metrics.ndcg_at_10 - dense_metrics.ndcg_at_10,
        recall_at_5_bootstrap_ci_lower=bootstrap_lower,
        recall_at_5_bootstrap_ci_upper=bootstrap_upper,
        bootstrap_samples=10_000,
        p95_latency_absolute_ms=p95_delta,
        p95_latency_relative_percent=_relative(p95_delta, dense.latency.p95_ms),
        tradeoff=(
            "Final routing improved retrieval quality but increased latency versus dense retrieval."
            if recall_delta > 0 and p95_delta > 0
            else "See numeric deltas; no qualitative benefit is implied."
        ),
    )
    answer_failures = _answer_failures(answer_report, answer_cases)
    reranking_latency = hybrid_reranker.stage_latencies.get(
        "reranking", LatencySummary(mean_ms=0, p50_ms=0, p95_ms=0)
    )
    return FinalBenchmarkReport(
        generated_at=datetime.now(UTC),
        benchmark_version=(
            retrieval_cases[0].benchmark_version
            if retrieval_cases and retrieval_cases[0].benchmark_version
            else "benchmark_v1"
        ),
        category_distribution=dict(
            sorted(Counter(case.category or "uncategorized" for case in retrieval_cases).items())
        )
        if retrieval_cases
        else {},
        methodology={
            "retrieval_questions": len(dense.queries),
            "answer_questions": answer_report.query_count,
            "max_cutoff": 10,
            "recall_definition": "query-level hit rate: at least one relevant chunk in top K",
            "latency_warmup": True,
            "same_dataset_for_all_retrievers": True,
            "final_pipeline": final_pipeline.experiment_name,
        },
        corpus=corpus,
        dense=dense,
        hybrid=hybrid,
        hybrid_reranker=hybrid_reranker,
        final_pipeline=final_pipeline,
        improvements=improvements,
        answer_evaluation=answer_report,
        answer_failures=answer_failures,
        system=SystemMeasurements(
            indexing_time_ms=indexing_time_ms,
            ingestion_time_ms=None,
            answer_latency=answer_report.latency,
            reranking_latency=reranking_latency,
            llm_token_usage=None,
            llm_cost_usd=None,
        ),
    )


def write_final_artifacts(report: FinalBenchmarkReport, experiments_dir: Path) -> None:
    """Persist the canonical report and each retrieval experiment separately."""

    experiments_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "final_dense.json": report.dense,
        "final_hybrid.json": report.hybrid,
        "final_hybrid_reranker.json": report.hybrid_reranker,
        "final_pipeline.json": report.final_pipeline,
        "final_benchmark.json": report,
    }
    for name, model in artifacts.items():
        path = experiments_dir / name
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        temporary.write_text(
            json.dumps(model.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)


def verify_final_artifact(
    path: Path,
    *,
    corpus_fingerprint: str,
    corpus: tuple[DocumentChunk, ...],
    retrieval_cases: list[EvaluationCase],
    answer_cases: list[AnswerEvaluationCase],
) -> FinalBenchmarkReport:
    """Reload an artifact and recompute ranking metrics from retained query results."""

    try:
        report = FinalBenchmarkReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as exc:
        raise ValueError(f"invalid final benchmark artifact: {path}") from exc
    expected_dataset = _dataset_fingerprint(retrieval_cases)
    expected_answer_dataset = _answer_dataset_fingerprint(answer_cases)
    relevant_counts = {
        case.question: sum(case.is_relevant(chunk) for chunk in corpus) for case in retrieval_cases
    }
    chunks_by_id = {chunk.chunk_id: chunk for chunk in corpus}
    cases_by_question = {case.question: case for case in retrieval_cases}
    experiments = (
        report.dense,
        report.hybrid,
        report.hybrid_reranker,
        report.final_pipeline,
    )
    for experiment in experiments:
        if experiment.corpus_fingerprint != corpus_fingerprint:
            raise ValueError(f"corpus fingerprint mismatch in {experiment.experiment_name}")
        if experiment.dataset_fingerprint != expected_dataset:
            raise ValueError(f"dataset fingerprint mismatch in {experiment.experiment_name}")
        if experiment.query_count != len(retrieval_cases) or len(experiment.queries) != len(
            retrieval_cases
        ):
            raise ValueError(f"query count mismatch in {experiment.experiment_name}")
        if [query.question for query in experiment.queries] != [
            case.question for case in retrieval_cases
        ]:
            raise ValueError(f"query order mismatch in {experiment.experiment_name}")
        if any(
            chunk_id not in chunks_by_id
            for query in experiment.queries
            for chunk_id in query.retrieved_chunk_ids
        ):
            raise ValueError(f"unknown retrieved chunk in {experiment.experiment_name}")
        for query in experiment.queries:
            case = cases_by_question[query.question]
            expected_relevance = tuple(
                case.is_relevant(chunks_by_id[chunk_id]) for chunk_id in query.retrieved_chunk_ids
            )
            expected_rank = next(
                (rank for rank, relevant in enumerate(expected_relevance, start=1) if relevant),
                None,
            )
            if query.relevance != expected_relevance or query.first_relevant_rank != expected_rank:
                raise ValueError(
                    f"stored relevance does not reproduce in {experiment.experiment_name}"
                )
        recalculated = _metrics(
            list(experiment.queries),
            relevant_counts,
            top_k=experiment.retrieval_cutoff,
        )
        if any(
            not math.isclose(
                float(experiment.metrics.model_dump()[name]),
                float(recalculated.model_dump()[name]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for name in type(recalculated).model_fields
        ):
            raise ValueError(f"stored metrics do not reproduce for {experiment.experiment_name}")
    if report.answer_evaluation.dataset_fingerprint != expected_answer_dataset:
        raise ValueError("answer dataset fingerprint mismatch")
    if report.corpus.chunk_count != len(corpus):
        raise ValueError("stored corpus chunk count mismatch")
    if report.answer_evaluation.query_count != len(answer_cases) or len(
        report.answer_evaluation.queries
    ) != len(answer_cases):
        raise ValueError("answer query count mismatch")
    if [query.question for query in report.answer_evaluation.queries] != [
        case.question for case in answer_cases
    ]:
        raise ValueError("answer query order mismatch")
    return report


def failure_breakdown(
    failures: tuple[FinalFailure, ...],
) -> dict[FinalFailureCategory, tuple[int, float]]:
    """Return counts and shares among objectively classified failures."""

    counts = Counter(failure.category for failure in failures)
    total = sum(counts.values())
    return {
        category: (count, count / total if total else 0.0)
        for category, count in sorted(counts.items())
    }


def _metrics(
    queries: list[FinalQueryEvaluation],
    relevant_counts: dict[str, int],
    *,
    top_k: int,
) -> FinalRetrievalMetrics:
    def hit_at(query: FinalQueryEvaluation, cutoff: int) -> float:
        return float(any(query.relevance[:cutoff]))

    precision_at_5 = [sum(query.relevance[:5]) / 5 for query in queries]
    reciprocal_ranks = [
        1 / query.first_relevant_rank if query.first_relevant_rank is not None else 0.0
        for query in queries
    ]
    ndcg_values = []
    for query in queries:
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, relevant in enumerate(query.relevance[:top_k], start=1)
            if relevant
        )
        ideal_count = min(relevant_counts[query.question], top_k)
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ndcg_values.append(dcg / ideal if ideal else 0.0)
    return FinalRetrievalMetrics(
        recall_at_1=float(np.mean([hit_at(query, 1) for query in queries])),
        recall_at_5=float(np.mean([hit_at(query, 5) for query in queries])),
        recall_at_10=float(np.mean([hit_at(query, 10) for query in queries])),
        precision_at_5=float(np.mean(precision_at_5)),
        mrr=float(np.mean(reciprocal_ranks)),
        ndcg_at_10=float(np.mean(ndcg_values)),
    )


def _classify_retrieval_failure(
    experiment: str,
    case: EvaluationCase,
    response: SearchResponse,
) -> FinalFailure:
    if not response.results:
        category: FinalFailureCategory = "RETRIEVAL_MISS"
        details = "retriever returned no candidates"
    else:
        metadata = response.results[0].chunk.metadata
        if case.expected_company and metadata.company != case.expected_company:
            category = "WRONG_COMPANY"
            details = f"top result company was {metadata.company}"
        elif case.expected_period and metadata.report_date != case.expected_period:
            category = "WRONG_PERIOD"
            details = f"top result period was {metadata.report_date}"
        elif (case.expected_filing and metadata.filing_type != case.expected_filing) or (
            case.expected_accession_number
            and metadata.accession_number != case.expected_accession_number
        ):
            category = "WRONG_FILING"
            details = f"top result accession was {metadata.accession_number}"
        elif case.expected_section and not metadata.section.casefold().startswith(
            case.expected_section.casefold()
        ):
            category = "WRONG_SECTION"
            details = f"top result section was {metadata.section}"
        else:
            category = "RETRIEVAL_MISS"
            details = "no labeled relevant chunk appeared in top 10"
    return FinalFailure(
        experiment=experiment,
        category=category,
        question=case.question,
        details=details,
    )


def _answer_failures(
    report: AnswerEvaluationReport,
    cases: list[AnswerEvaluationCase],
) -> tuple[FinalFailure, ...]:
    failures = []
    for result, case in zip(report.queries, cases, strict=True):
        if result.passed:
            continue
        if result.actual_status is None:
            category: FinalFailureCategory = "GENERATION_ERROR"
            details = "answer pipeline raised an exception"
        elif case.expected_status == "answered" and result.actual_status == "insufficient_evidence":
            category = "INSUFFICIENT_CONTEXT"
            details = "answer was refused despite labeled evidence"
        elif case.expected_status == "insufficient_evidence" and result.actual_status == "answered":
            category = "UNSUPPORTED_ANSWER"
            details = "answer was produced for an unsupported question"
        elif not result.checks.get("values", True):
            category = "NUMERIC_REASONING_ERROR"
            details = "deterministic numeric result did not match ground truth"
        else:
            category = "GENERATION_ERROR"
            details = "route, citation, grounding, or evidence validation failed"
        failures.append(
            FinalFailure(
                experiment="Answer pipeline",
                category=category,
                question=result.question,
                details=details,
            )
        )
    return tuple(failures)


def _latency(values: list[float]) -> LatencySummary:
    samples = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        mean_ms=float(np.mean(samples)),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
    )


def _stage_latencies(
    queries: list[FinalQueryEvaluation],
) -> dict[str, LatencySummary]:
    names = sorted({name for query in queries for name in query.stage_latencies_ms})
    return {
        name: _latency(
            [
                query.stage_latencies_ms[name]
                for query in queries
                if name in query.stage_latencies_ms
            ]
        )
        for name in names
    }


def _dataset_fingerprint(cases: list[EvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        payload = case.model_dump(mode="json")
        payload["relevant_chunk_ids"] = sorted(case.relevant_chunk_ids)
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _answer_dataset_fingerprint(cases: list[AnswerEvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _tree_bytes(path: Path) -> int:
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def _relative(delta: float, baseline: float) -> float | None:
    return delta / baseline * 100 if baseline else None


def _bootstrap_recall_delta(
    dense: FinalRetrievalExperiment,
    final: FinalRetrievalExperiment,
    *,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return a deterministic paired percentile CI for the Recall@5 difference."""

    if len(dense.queries) != len(final.queries) or not dense.queries:
        raise ValueError("paired bootstrap requires aligned non-empty query results")
    dense_hits = np.asarray([any(query.relevance[:5]) for query in dense.queries], dtype=np.float64)
    final_hits = np.asarray([any(query.relevance[:5]) for query in final.queries], dtype=np.float64)
    paired_delta = final_hits - dense_hits
    generator = np.random.default_rng(20261005)
    indices = generator.integers(0, len(paired_delta), size=(samples, len(paired_delta)))
    distribution = paired_delta[indices].mean(axis=1)
    lower, upper = np.percentile(distribution, [2.5, 97.5])
    return float(lower), float(upper)
