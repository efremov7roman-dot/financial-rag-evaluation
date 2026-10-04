from pathlib import Path

from financial_rag.evaluation.models import EvaluationCase
from financial_rag.evaluation.retrieval import evaluate_retriever
from financial_rag.retrieval.dense import DenseRetriever
from tests.test_dense_retrieval import KeywordEmbedder, build_test_index


def test_comparable_evaluation_reports_precision_and_ndcg(tmp_path: Path) -> None:
    index, _ = build_test_index(tmp_path)
    retriever = DenseRetriever(index, KeywordEmbedder())
    cases = [
        EvaluationCase(question="revenue", relevant_chunk_ids=frozenset({"a" * 64})),
        EvaluationCase(question="risk", relevant_chunk_ids=frozenset({"b" * 64})),
    ]

    report = evaluate_retriever(
        experiment_name="test dense",
        configuration={"top_k": 1},
        model_name=KeywordEmbedder().model_name,
        corpus_fingerprint=index.manifest.corpus_fingerprint,
        corpus=index.chunks,
        retriever=retriever,
        cases=cases,
        top_k=1,
    )

    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 1.0
    assert report.mrr == 1.0
    assert report.ndcg_at_k == 1.0
    assert report.stage_latencies["dense"].p95_ms >= 0
