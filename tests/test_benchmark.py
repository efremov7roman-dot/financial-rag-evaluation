from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from financial_rag.evaluation.answers import evaluate_answers, load_answer_cases
from financial_rag.evaluation.benchmark import build_benchmark_report, write_benchmark_artifacts
from financial_rag.evaluation.dense import load_evaluation_cases
from financial_rag.evaluation.models import (
    AnswerEvaluationCase,
    AnswerEvaluationReport,
    AnswerQueryEvaluation,
    LatencySummary,
    RankedQueryEvaluation,
    RetrievalEvaluationReport,
)
from financial_rag.intelligence.engine import FinancialIntelligenceEngine
from financial_rag.intelligence.routing import QueryRouter
from financial_rag.intelligence.xbrl import InlineXBRLStore
from tests.test_xbrl_intelligence import EmptyRetriever, _chunk, _fact


def _engine() -> FinancialIntelligenceEngine:
    chunk = _chunk()
    return FinancialIntelligenceEngine(
        QueryRouter((chunk,)),
        EmptyRetriever(),
        InlineXBRLStore((_fact(2023, "100"), _fact(2024, "125"))),
    )


def test_answer_evaluation_covers_values_citations_and_refusal() -> None:
    cases = [
        AnswerEvaluationCase(
            question="What was EXM revenue growth from 2023 to 2024?",
            expected_status="answered",
            expected_route="calculation",
            expected_values={"result": Decimal("25")},
            minimum_citations=2,
            expected_company="Example Corp",
            expected_period_years=(2023, 2024),
        ),
        AnswerEvaluationCase(
            question="What was EXM revenue in 1900?",
            expected_status="insufficient_evidence",
            expected_route="structured_fact",
        ),
    ]

    facts = (_fact(2023, "100"), _fact(2024, "125"))
    report, failures = evaluate_answers(
        _engine(),
        cases,
        corpus=(_chunk(),),
        facts=facts,
    )

    assert report.pass_rate == 1
    assert report.value_accuracy == 1
    assert report.grounded_answer_rate == 1
    assert report.evidence_accuracy == 1
    assert report.refusal_accuracy == 1
    assert failures == ()


def test_answer_evaluation_classifies_wrong_calculation() -> None:
    case = AnswerEvaluationCase(
        question="What was EXM revenue growth from 2023 to 2024?",
        expected_status="answered",
        expected_route="calculation",
        expected_values={"result": Decimal("99")},
        minimum_citations=2,
    )

    report, failures = evaluate_answers(_engine(), [case])

    assert report.pass_rate == 0
    assert [failure.category for failure in failures] == ["calculation_error"]


def test_benchmark_builds_failure_report_and_artifacts(tmp_path: Path) -> None:
    retrieval = RetrievalEvaluationReport(
        experiment_name="test retrieval",
        configuration={"top_k": 5},
        model_name="test/model",
        corpus_fingerprint="a" * 64,
        dataset_fingerprint="b" * 64,
        evaluated_at=datetime.now(UTC),
        corpus_document_count=2,
        query_count=1,
        top_k=5,
        recall_at_k=0,
        precision_at_k=0,
        mrr=0,
        ndcg_at_k=0,
        latency=LatencySummary(mean_ms=1, p50_ms=1, p95_ms=1),
        stage_latencies={},
        queries=(
            RankedQueryEvaluation(
                question="missing",
                retrieved_chunk_ids=(),
                relevance=(),
                first_relevant_rank=None,
                recall=0,
                precision=0,
                reciprocal_rank=0,
                ndcg=0,
                latency_ms=1,
                stage_latencies_ms={},
            ),
        ),
    )
    answers = AnswerEvaluationReport(
        dataset_fingerprint="c" * 64,
        evaluated_at=datetime.now(UTC),
        query_count=1,
        pass_rate=1,
        status_accuracy=1,
        route_accuracy=1,
        value_accuracy=1,
        citation_accuracy=1,
        grounded_answer_rate=1,
        evidence_accuracy=1,
        refusal_accuracy=1,
        latency=LatencySummary(mean_ms=1, p50_ms=1, p95_ms=1),
        queries=(
            AnswerQueryEvaluation(
                question="answer",
                passed=True,
                checks={"status": True},
                actual_status="answered",
                actual_route="structured_fact",
                citation_count=1,
                latency_ms=1,
            ),
        ),
    )

    report = build_benchmark_report((retrieval,), answers, ())
    json_path = tmp_path / "benchmark.json"
    markdown_path = tmp_path / "benchmark.md"
    write_benchmark_artifacts(report, json_path, markdown_path)

    assert report.failures[0].category == "retrieval_miss"
    assert "Retrieval comparison" in markdown_path.read_text(encoding="utf-8")
    assert '"retrieval_miss"' in json_path.read_text(encoding="utf-8")


def test_phase5_datasets_are_larger_and_valid() -> None:
    retrieval = load_evaluation_cases(Path("data/eval/retrieval_benchmark.jsonl"))
    answers = load_answer_cases(Path("data/eval/answer_benchmark.jsonl"))

    assert len(retrieval) == 18
    assert len(answers) == 10
