from pathlib import Path

import pytest

from financial_rag.evaluation.dense import evaluate_dense, load_evaluation_cases
from financial_rag.evaluation.models import EvaluationCase
from tests.test_dense_retrieval import KeywordEmbedder, build_test_index


def test_dense_evaluation_measures_recall_mrr_and_latency(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)
    cases = [
        EvaluationCase(question="revenue performance", relevant_chunk_ids=frozenset({"a" * 64})),
        EvaluationCase(question="business risks", relevant_chunk_ids=frozenset({"b" * 64})),
    ]

    report = evaluate_dense(index, KeywordEmbedder(), cases, top_k=1)

    assert report.recall_at_k == 1.0
    assert report.mrr == 1.0
    assert report.query_count == 2
    assert report.p50_latency_ms >= 0
    assert report.p95_latency_ms >= report.p50_latency_ms


def test_load_real_baseline_dataset() -> None:
    cases = load_evaluation_cases(Path("data/eval/dense_baseline.jsonl"))

    assert len(cases) == 4
    assert cases[0].expected_accession_number == "0000320193-24-000123"


def test_evaluation_case_requires_judgment() -> None:
    with pytest.raises(ValueError, match="relevance judgment"):
        EvaluationCase(question="unlabeled")
