"""Unified benchmark assembly, failure analysis, and artifact rendering."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from financial_rag.evaluation.models import (
    AnswerEvaluationReport,
    BenchmarkReport,
    BenchmarkSummary,
    FailureRecord,
    RetrievalComparisonReport,
    RetrievalEvaluationReport,
)


def build_benchmark_report(
    retrieval_reports: tuple[RetrievalEvaluationReport, ...],
    answers: AnswerEvaluationReport,
    answer_failures: tuple[FailureRecord, ...],
    *,
    acceptable_rank: int = 3,
) -> BenchmarkReport:
    """Combine comparable experiments and classify retrieval failures."""

    if not retrieval_reports:
        raise ValueError("benchmark requires at least one retrieval report")
    retrieval_failures = tuple(
        failure
        for report in retrieval_reports
        for failure in _retrieval_failures(report, acceptable_rank=acceptable_rank)
    )
    failures = (*retrieval_failures, *answer_failures)
    winner = max(
        retrieval_reports,
        key=lambda report: (report.ndcg_at_k, report.mrr, -report.latency.mean_ms),
    )
    return BenchmarkReport(
        generated_at=datetime.now(UTC),
        summary=BenchmarkSummary(
            retrieval_winner=winner.experiment_name,
            retrieval_recall_at_k=winner.recall_at_k,
            retrieval_mrr=winner.mrr,
            retrieval_ndcg_at_k=winner.ndcg_at_k,
            answer_pass_rate=answers.pass_rate,
            failure_count=len(failures),
        ),
        retrieval=RetrievalComparisonReport(experiments=retrieval_reports),
        answers=answers,
        failures=failures,
    )


def write_benchmark_artifacts(
    report: BenchmarkReport, json_path: Path, markdown_path: Path
) -> None:
    """Write atomic machine-readable and human-readable benchmark artifacts."""

    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(
        json_path,
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
    )
    _atomic_write(markdown_path, render_benchmark_markdown(report))


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    """Render comparison tables and failure counts from the JSON report."""

    lines = [
        "# PHASE 5 benchmark",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Retrieval comparison",
        "",
        "| Experiment | Recall@K | Precision@K | MRR | nDCG@K | Mean ms | p95 ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in report.retrieval.experiments:
        lines.append(
            f"| {experiment.experiment_name} | {experiment.recall_at_k:.3f} | "
            f"{experiment.precision_at_k:.3f} | {experiment.mrr:.3f} | "
            f"{experiment.ndcg_at_k:.3f} | {experiment.latency.mean_ms:.2f} | "
            f"{experiment.latency.p95_ms:.2f} |"
        )
    answers = report.answers
    lines.extend(
        [
            "",
            "## Evidence-grounded answer evaluation",
            "",
            (
                "| Pass rate | Status | Route | Values | Citations | Grounded | Evidence | "
                "Refusals | Mean ms |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            f"| {answers.pass_rate:.3f} | {answers.status_accuracy:.3f} | "
            f"{answers.route_accuracy:.3f} | {answers.value_accuracy:.3f} | "
            f"{answers.citation_accuracy:.3f} | {answers.grounded_answer_rate:.3f} | "
            f"{answers.evidence_accuracy:.3f} | {answers.refusal_accuracy:.3f} | "
            f"{answers.latency.mean_ms:.2f} |",
            "",
            "## Failure taxonomy",
            "",
            "| Category | Count |",
            "| --- | ---: |",
        ]
    )
    counts = Counter(failure.category for failure in report.failures)
    if counts:
        lines.extend(f"| {category} | {count} |" for category, count in sorted(counts.items()))
    else:
        lines.append("| none | 0 |")
    lines.extend(
        [
            "",
            f"Winner: **{report.summary.retrieval_winner}**. "
            f"Total classified failures: **{report.summary.failure_count}**.",
            "",
        ]
    )
    return "\n".join(lines)


def _retrieval_failures(
    report: RetrievalEvaluationReport, *, acceptable_rank: int
) -> tuple[FailureRecord, ...]:
    failures = []
    for query in report.queries:
        if query.first_relevant_rank is None:
            failures.append(
                FailureRecord(
                    component="retrieval",
                    category="retrieval_miss",
                    experiment=report.experiment_name,
                    question=query.question,
                    details=f"no relevant result in top {report.top_k}",
                )
            )
        elif query.first_relevant_rank > acceptable_rank:
            failures.append(
                FailureRecord(
                    component="retrieval",
                    category="ranking_failure",
                    experiment=report.experiment_name,
                    question=query.question,
                    details=(
                        f"first relevant result ranked {query.first_relevant_rank}; "
                        f"target rank <= {acceptable_rank}"
                    ),
                )
            )
    return tuple(failures)


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
