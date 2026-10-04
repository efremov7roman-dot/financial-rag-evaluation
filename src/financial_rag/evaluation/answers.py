"""Evaluation of deterministic finance answers, citations, and refusal behavior."""

import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from financial_rag.errors import RetrievalError
from financial_rag.evaluation.models import (
    AnswerEvaluationCase,
    AnswerEvaluationReport,
    AnswerQueryEvaluation,
    FailureCategory,
    FailureRecord,
    LatencySummary,
)
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.intelligence.engine import FinancialIntelligenceEngine
from financial_rag.intelligence.models import EvidenceCitation, XBRLFact


def load_answer_cases(path: Path) -> list[AnswerEvaluationCase]:
    """Load a non-empty JSONL answer benchmark with line context."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RetrievalError(f"failed to read answer evaluation dataset {path}") from exc
    cases: list[AnswerEvaluationCase] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            cases.append(AnswerEvaluationCase.model_validate_json(line))
        except ValidationError as exc:
            raise RetrievalError(f"invalid answer case at {path}:{line_number}") from exc
    if not cases:
        raise RetrievalError(f"answer evaluation dataset is empty: {path}")
    return cases


def evaluate_answers(
    engine: FinancialIntelligenceEngine,
    cases: list[AnswerEvaluationCase],
    *,
    corpus: tuple[DocumentChunk, ...] = (),
    facts: tuple[XBRLFact, ...] = (),
) -> tuple[AnswerEvaluationReport, tuple[FailureRecord, ...]]:
    """Evaluate routing, values, citations, grounding, refusals, and latency."""

    if not cases:
        raise ValueError("answer evaluation requires at least one case")
    results: list[AnswerQueryEvaluation] = []
    failures: list[FailureRecord] = []
    value_results: list[bool] = []
    refusal_results: list[bool] = []
    for case in cases:
        started_at = time.perf_counter()
        try:
            answer = engine.answer(case.question)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started_at) * 1_000
            results.append(
                AnswerQueryEvaluation(
                    question=case.question,
                    passed=False,
                    checks={"runtime": False},
                    citation_count=0,
                    latency_ms=latency_ms,
                )
            )
            failures.append(
                FailureRecord(
                    component="answer",
                    category="runtime_error",
                    question=case.question,
                    details=f"{type(exc).__name__}: {exc}",
                )
            )
            if case.expected_values:
                value_results.append(False)
            if case.expected_status == "insufficient_evidence":
                refusal_results.append(False)
            continue

        latency_ms = (time.perf_counter() - started_at) * 1_000
        checks = {
            "status": answer.status == case.expected_status,
            "route": answer.route == case.expected_route,
            "values": all(
                key in answer.values
                and abs(answer.values[key] - expected) <= case.absolute_tolerance
                for key, expected in case.expected_values.items()
            ),
            "citations": len(answer.citations) >= case.minimum_citations,
            "grounding": answer.status != "answered" or bool(answer.citations),
            "evidence": _evidence_is_valid(
                answer.citations,
                case,
                corpus=corpus,
                facts=facts,
            ),
            "refusal": (
                answer.status == "insufficient_evidence" and not answer.citations
                if case.expected_status == "insufficient_evidence"
                else True
            ),
        }
        results.append(
            AnswerQueryEvaluation(
                question=case.question,
                passed=all(checks.values()),
                checks=checks,
                actual_status=answer.status,
                actual_route=answer.route,
                actual_values=answer.values,
                citation_count=len(answer.citations),
                latency_ms=latency_ms,
            )
        )
        if case.expected_values:
            value_results.append(checks["values"])
        if case.expected_status == "insufficient_evidence":
            refusal_results.append(checks["refusal"])
        failures.extend(_answer_failures(case, checks))

    latencies = [result.latency_ms for result in results]
    report = AnswerEvaluationReport(
        dataset_fingerprint=_fingerprint(cases),
        evaluated_at=datetime.now(UTC),
        query_count=len(results),
        pass_rate=_mean([result.passed for result in results]),
        status_accuracy=_mean([result.checks.get("status", False) for result in results]),
        route_accuracy=_mean([result.checks.get("route", False) for result in results]),
        value_accuracy=_mean(value_results),
        citation_accuracy=_mean([result.checks.get("citations", False) for result in results]),
        grounded_answer_rate=_mean([result.checks.get("grounding", False) for result in results]),
        evidence_accuracy=_mean([result.checks.get("evidence", False) for result in results]),
        refusal_accuracy=_mean(refusal_results),
        latency=_latency(latencies),
        queries=tuple(results),
    )
    return report, tuple(failures)


def _answer_failures(case: AnswerEvaluationCase, checks: dict[str, bool]) -> list[FailureRecord]:
    categories: list[tuple[FailureCategory, str]] = []
    if not checks["route"]:
        categories.append(("wrong_route", "query was assigned to the wrong answer route"))
    if not checks["status"]:
        status_category: FailureCategory = (
            "refusal_error"
            if case.expected_status == "insufficient_evidence"
            else "missing_evidence"
        )
        categories.append(
            (status_category, "answer status did not match the expected evidence behavior")
        )
    if not checks["values"]:
        categories.append(
            ("calculation_error", "one or more expected numeric values were incorrect")
        )
    if not checks["citations"] or not checks["grounding"] or not checks["evidence"]:
        categories.append(
            ("citation_failure", "answer did not provide the required evidence citations")
        )
    if not checks["refusal"]:
        categories.append(("refusal_error", "unsupported question was not refused cleanly"))
    return [
        FailureRecord(
            component="answer", category=category, question=case.question, details=details
        )
        for category, details in categories
    ]


def _mean(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 1.0


def _latency(values: list[float]) -> LatencySummary:
    samples = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        mean_ms=float(np.mean(samples)),
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
    )


def _fingerprint(cases: list[AnswerEvaluationCase]) -> str:
    digest = hashlib.sha256()
    for case in cases:
        digest.update(case.model_dump_json().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _evidence_is_valid(
    citations: tuple[EvidenceCitation, ...],
    case: AnswerEvaluationCase,
    *,
    corpus: tuple[DocumentChunk, ...],
    facts: tuple[XBRLFact, ...],
) -> bool:
    if case.expected_status == "insufficient_evidence":
        return not citations
    if not citations:
        return False
    if not corpus and not facts:
        return True
    mapped = [_map_citation(citation, corpus=corpus, facts=facts) for citation in citations]
    if any(item is None for item in mapped):
        return False
    evidence = [item for item in mapped if item is not None]
    if case.expected_company and any(company != case.expected_company for company, _ in evidence):
        return False
    years = {year for _, year in evidence if year is not None}
    return not case.expected_period_years or set(case.expected_period_years) <= years


def _map_citation(
    citation: EvidenceCitation,
    *,
    corpus: tuple[DocumentChunk, ...],
    facts: tuple[XBRLFact, ...],
) -> tuple[str, int | None] | None:
    if citation.locator.startswith("xbrl="):
        for fact in facts:
            locator = f"xbrl={fact.concept};period={fact.period_end.isoformat()}"
            excerpt = f"{fact.concept}={fact.value} {fact.unit}"
            if (
                citation.locator == locator
                and citation.excerpt == excerpt
                and citation.accession_number == fact.accession_number
                and citation.source_url == fact.source_url
            ):
                return fact.company, fact.period_end.year
        return None
    chunk_id = citation.locator.rpartition("chunk=")[2]
    for chunk in corpus:
        if (
            chunk.chunk_id == chunk_id
            and citation.excerpt == chunk.text[:300]
            and citation.accession_number == chunk.metadata.accession_number
            and citation.source_url == chunk.metadata.source_url
        ):
            year = chunk.metadata.report_date.year if chunk.metadata.report_date else None
            return chunk.metadata.company, year
    return None
