"""Human-readable final benchmark and resume metrics generated from the canonical JSON model."""

from collections import Counter
from pathlib import Path
from typing import get_args

from financial_rag.evaluation.final_models import (
    FinalBenchmarkReport,
    FinalFailure,
    FinalFailureCategory,
    FinalRetrievalExperiment,
)


def write_final_documents(
    report: FinalBenchmarkReport, benchmark_path: Path, resume_path: Path
) -> None:
    """Atomically render both human-facing documents from measured fields."""

    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(benchmark_path, render_benchmark(report))
    _atomic_write(resume_path, render_resume_metrics(report))


def render_benchmark(report: FinalBenchmarkReport) -> str:
    corpus = report.corpus
    experiments = (
        report.dense,
        report.hybrid,
        report.hybrid_reranker,
        report.final_pipeline,
    )
    lines = [
        "# Benchmark V2",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        "",
        "## Dataset and corpus",
        "",
        (
            f"The benchmark uses {corpus.retrieval_question_count} retrieval questions and "
            f"{corpus.answer_question_count} answer questions over {corpus.filing_count} real SEC "
            f"filings ({corpus.ten_k_count} 10-K and {corpus.ten_q_count} 10-Q) from "
            f"{corpus.company_count} companies ({', '.join(corpus.companies)}). The filings span "
            "report years "
            f"{corpus.report_year_min}-{corpus.report_year_max} and contain "
            f"{corpus.processed_document_count} processed documents, {corpus.chunk_count} chunks, "
            f"and {corpus.xbrl_fact_count} normalized XBRL facts. Filing dates run from "
            f"{corpus.filing_date_min.isoformat()} to {corpus.filing_date_max.isoformat()}."
        ),
        "",
        (
            f"Raw filings occupy {_bytes(corpus.raw_corpus_bytes)}, processed chunks "
            f"{_bytes(corpus.processed_corpus_bytes)}, for {_bytes(corpus.total_corpus_bytes)} "
            f"of corpus data in total. The dense index occupies "
            f"{_bytes(corpus.dense_index_bytes)}. The vector matrix contains "
            f"{corpus.vector_count} x {corpus.vector_dimension} dimensions and occupies "
            f"{_bytes(corpus.vector_file_bytes)}."
        ),
        "",
        "The frozen benchmark was constructed before retrieval runs. Every retrieval question has "
        "company, filing type, reporting period, accession, section, source URL, evidence excerpt, "
        "and exact relevant chunk IDs. Numerical cases use SEC inline-XBRL source facts and "
        "deterministic Decimal calculations.",
        "",
        "### Question categories",
        "",
        "| Category | Questions |",
        "| --- | ---: |",
        *[
            f"| {category} | {count} |"
            for category, count in sorted(report.category_distribution.items())
        ],
        "",
        "## Methodology",
        "",
        "Each retriever receives the identical ordered dataset and is warmed once before timing. "
        "One top-10 call supplies Recall@1/5/10, Precision@5, MRR, and nDCG@10. Recall@K is the "
        "query-level hit rate: a question scores 1 when at least one labeled chunk is in the first "
        "K results. Latencies are local CPU measurements and include query embedding; reranker "
        "latency is also recorded separately.",
        "A deterministic 10,000-sample paired bootstrap estimates the 95% confidence interval for "
        "the Final-minus-Dense Recall@5 difference.",
        "",
        "## Retrieval results",
        "",
        "| Pipeline | Recall@1 | Recall@5 | Recall@10 | Precision@5 | MRR | nDCG@10 | p50 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in experiments:
        metrics = experiment.metrics
        lines.append(
            f"| {experiment.experiment_name} | {metrics.recall_at_1:.3f} | "
            f"{metrics.recall_at_5:.3f} | {metrics.recall_at_10:.3f} | "
            f"{metrics.precision_at_5:.3f} | {metrics.mrr:.3f} | "
            f"{metrics.ndcg_at_10:.3f} | {experiment.latency.p50_ms:.2f} ms | "
            f"{experiment.latency.p95_ms:.2f} ms |"
        )
    improvements = report.improvements
    relative_recall = _percent(improvements.recall_at_5_relative_percent)
    relative_mrr = _percent(improvements.mrr_relative_percent)
    relative_p95 = _percent(improvements.p95_latency_relative_percent)
    lines.extend(
        [
            "",
            "## Measured improvement and trade-off",
            "",
            (
                f"Final versus Dense: Recall@5 changed by "
                f"{improvements.recall_at_5_absolute:+.3f} ({relative_recall}), MRR by "
                f"{improvements.mrr_absolute:+.3f} ({relative_mrr}), and nDCG@10 by "
                f"{improvements.ndcg_at_10_absolute:+.3f}. p95 latency changed by "
                f"{improvements.p95_latency_absolute_ms:+.2f} ms ({relative_p95})."
            ),
            (
                f"Paired bootstrap Recall@5 difference: observed "
                f"{improvements.recall_at_5_absolute:+.3f}, 95% CI "
                f"[{improvements.recall_at_5_bootstrap_ci_lower:+.3f}, "
                f"{improvements.recall_at_5_bootstrap_ci_upper:+.3f}] "
                f"({improvements.bootstrap_samples:,} samples)."
            ),
            "",
            improvements.tradeoff,
            "",
            "The final pipeline is routed Hybrid RRF without the cross-encoder. This is an "
            "evidence-based choice: the reranker configuration is measured separately and is not "
            "claimed as part of production when its quality/latency trade-off is inferior.",
            "",
            "## Failure analysis",
            "",
        ]
    )
    for experiment in experiments:
        lines.extend(_failure_table(experiment))
    lines.extend(_failure_lines("Answer pipeline", report.answer_failures))
    final_failures = (*report.final_pipeline.failures, *report.answer_failures)
    top_causes = Counter(failure.category for failure in final_failures).most_common(3)
    cause_lines = [
        f"- {category}: {count}/{len(final_failures)} failures ({count / len(final_failures):.1%})."
        for category, count in top_causes
    ]
    if final_failures and len(cause_lines) < 3:
        cause_lines.append("- No third non-zero failure category was observed.")
    lines.extend(
        [
            "### Top three Final/answer failure causes",
            "",
            *(cause_lines if final_failures else ["No Final/answer failures were observed."]),
            "",
        ]
    )
    answer = report.answer_evaluation
    answer_passes = sum(query.passed for query in answer.queries)
    lines.extend(
        [
            "## Financial answers and evidence grounding",
            "",
            (
                "| Cases | Pass | Status | Route | Values | Citations | Grounded | Evidence | "
                "Refusal | p50 | p95 |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {answer.query_count} | {answer.pass_rate:.3f} | "
                f"{answer.status_accuracy:.3f} | {answer.route_accuracy:.3f} | "
                f"{answer.value_accuracy:.3f} | {answer.citation_accuracy:.3f} | "
                f"{answer.grounded_answer_rate:.3f} | {answer.evidence_accuracy:.3f} | "
                f"{answer.refusal_accuracy:.3f} | {answer.latency.p50_ms:.2f} ms | "
                f"{answer.latency.p95_ms:.2f} ms |"
            ),
            "",
            f"Deterministic labeled cases passed: {answer_passes}/{answer.query_count}.",
            "",
            "Citation validation resolves every cited accession and URL against the local corpus, "
            "checks company and period labels, and verifies either the exact XBRL fact or the "
            "cited "
            "chunk excerpt. No LLM is used, so token usage and LLM cost are not applicable.",
            "",
            "## System measurements",
            "",
            f"- Indexing time: {_milliseconds(report.system.indexing_time_ms)}",
            "- Ingestion time: not reported; the audit reused the immutable SEC cache and did not "
            "repeat network downloads.",
            (
                f"- Cross-encoder reranking p50/p95: "
                f"{report.system.reranking_latency.p50_ms:.2f}/"
                f"{report.system.reranking_latency.p95_ms:.2f} ms."
            ),
            (
                f"- Evidence-answer end-to-end p50/p95: "
                f"{report.system.answer_latency.p50_ms:.2f}/"
                f"{report.system.answer_latency.p95_ms:.2f} ms."
            ),
            "",
            "## Limitations",
            "",
            "- The corpus contains six large US public technology companies; results should not "
            "be generalized to all issuers, industries, filing languages, or filing types.",
            "- Section-level labels can mark multiple chunks relevant and do not measure passage "
            "completeness for long answers.",
            "- Metadata filtering is applied to a bounded candidate pool, so the correct period "
            "can "
            "still be absent before filtering.",
            "- Latency is a local CPU measurement and varies with host load despite model warm-up.",
            "- Narrative output returns evidence passages rather than LLM-generated synthesis.",
            "- XBRL answer accuracy is reported only for the labeled denominator above and is not "
            "a claim of universal financial-answer accuracy.",
            "",
            "## Benchmark V1 versus V2",
            "",
            "| Benchmark | Filings | Questions | Dense Recall@5 | Final Recall@5 | "
            "Dense MRR | Final MRR |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            "| V1 | 9 | 18 | 0.444 | 0.667 | 0.278 | 0.540 |",
            (
                f"| V2 | {corpus.filing_count} | {corpus.retrieval_question_count} | "
                f"{report.dense.metrics.recall_at_5:.3f} | "
                f"{report.final_pipeline.metrics.recall_at_5:.3f} | "
                f"{report.dense.metrics.mrr:.3f} | "
                f"{report.final_pipeline.metrics.mrr:.3f} |"
            ),
            "",
            (
                "On V2, the Final pipeline "
                + ("retained" if improvements.recall_at_5_absolute > 0 else "did not retain")
                + " a Recall@5 advantage over Dense."
            ),
            "",
            "## Reproduction",
            "",
            "From a fresh clone, first reconstruct and verify the exact public SEC corpus:",
            "",
            "```powershell",
            "financial-rag reproduce-benchmark-v2",
            "",
            "financial-rag run-final-benchmark --retrieval-dataset "
            "data/eval/benchmark_v2_retrieval.jsonl --answer-dataset "
            "data/eval/benchmark_v2_answers.jsonl --experiments-dir "
            "experiments/benchmark_v2 --skip-indexing",
            "```",
            "",
            "The exact accession, source, processing, and hash contract is checked in at "
            "`experiments/benchmark_v2/corpus_manifest.json`. Large SEC files, indexes, and "
            "model weights are generated locally and excluded from Git.",
            "",
            "The canonical machine-readable source is "
            "`experiments/benchmark_v2/final_benchmark.json`; the four "
            "pipeline artifacts retain every ranking and stage latency separately.",
            "",
        ]
    )
    return "\n".join(lines)


def render_resume_metrics(report: FinalBenchmarkReport) -> str:
    corpus = report.corpus
    dense = report.dense.metrics
    final = report.final_pipeline.metrics
    improvements = report.improvements
    return "\n".join(
        [
            "# Resume metrics",
            "",
            "All claims below are generated from `experiments/benchmark_v2/final_benchmark.json`.",
            "",
            (
                f"- Built an evaluation-first financial retrieval system over "
                f"{corpus.filing_count} real SEC filings ({corpus.ten_k_count} 10-K, "
                f"{corpus.ten_q_count} 10-Q), indexing {corpus.chunk_count} text chunks and "
                f"{corpus.xbrl_fact_count} normalized XBRL facts with source-level provenance."
            ),
            (
                f"- Improved Recall@5 from {dense.recall_at_5:.3f} to "
                f"{final.recall_at_5:.3f} ({improvements.recall_at_5_absolute:+.3f} absolute, "
                f"{_percent(improvements.recall_at_5_relative_percent)} relative) and MRR from "
                f"{dense.mrr:.3f} to {final.mrr:.3f} on a "
                f"{corpus.retrieval_question_count}-question temporal SEC benchmark using "
                f"metadata-routed Dense+BM25 RRF."
            ),
            (
                f"- Designed deterministic XBRL calculations and evidence validation that passed "
                f"{sum(query.passed for query in report.answer_evaluation.queries)}/"
                f"{report.answer_evaluation.query_count} "
                f"labeled answer cases with {report.answer_evaluation.evidence_accuracy:.3f} "
                f"citation-evidence accuracy; final retrieval p95 was "
                f"{report.final_pipeline.latency.p95_ms:.2f} ms on local CPU."
            ),
            "",
            "Do not combine these measurements with unmeasured business-impact claims.",
            "",
        ]
    )


def _failure_table(experiment: FinalRetrievalExperiment) -> list[str]:
    return _failure_lines(experiment.experiment_name, experiment.failures)


def _failure_lines(title: str, failures: tuple[FinalFailure, ...]) -> list[str]:
    counts = Counter(failure.category for failure in failures)
    total = sum(counts.values())
    lines = [f"### {title}", ""]
    lines.extend(["| Failure | Count | Share of failures |", "| --- | ---: | ---: |"])
    lines.extend(
        f"| {category} | {counts[category]} | {counts[category] / total if total else 0.0:.1%} |"
        for category in get_args(FinalFailureCategory)
    )
    lines.append("")
    return lines


def _bytes(value: int) -> str:
    return f"{value / (1024 * 1024):.2f} MiB ({value:,} bytes)"


def _milliseconds(value: float | None) -> str:
    return "not measured" if value is None else f"{value:.2f} ms"


def _percent(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.1f}%"


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
