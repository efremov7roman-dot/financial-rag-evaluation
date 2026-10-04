"""Command-line interface for project foundation commands."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, cast

import typer

from financial_rag import __version__
from financial_rag.config import Settings, get_settings
from financial_rag.embeddings import FastEmbedModel
from financial_rag.errors import FinancialRAGError, ReproductionError
from financial_rag.evaluation.answers import evaluate_answers, load_answer_cases
from financial_rag.evaluation.benchmark import build_benchmark_report, write_benchmark_artifacts
from financial_rag.evaluation.benchmark_v2 import freeze_benchmark_v2
from financial_rag.evaluation.dense import (
    evaluate_dense,
    load_evaluation_cases,
    write_evaluation_report,
)
from financial_rag.evaluation.final_audit import (
    build_final_report,
    classify_reranker_drops,
    evaluate_final_retriever,
    measure_corpus,
    verify_final_artifact,
    write_final_artifacts,
)
from financial_rag.evaluation.final_report import write_final_documents
from financial_rag.evaluation.models import (
    EvaluationCase,
    RetrievalComparisonReport,
    RetrievalEvaluationReport,
)
from financial_rag.evaluation.retrieval import evaluate_retriever, write_comparison_report
from financial_rag.ingestion.models import FilingType
from financial_rag.ingestion.pipeline import IngestionPipeline
from financial_rag.ingestion.sec_client import SECClient
from financial_rag.intelligence import (
    FinancialIntelligenceEngine,
    InlineXBRLStore,
    QueryRouter,
    RoutedRetriever,
)
from financial_rag.logging import configure_logging
from financial_rag.reproduction import (
    file_sha256,
    load_corpus_manifest,
    prepare_benchmark_corpus,
    text_sha256,
    verify_frozen_datasets,
)
from financial_rag.reranking import FastEmbedReranker, RerankingRetriever
from financial_rag.retrieval import DenseIndex, DenseIndexBuilder
from financial_rag.retrieval.bm25 import BM25Index
from financial_rag.retrieval.dense import DenseRetriever
from financial_rag.retrieval.hybrid import HybridRetriever

app = typer.Typer(
    name="financial-rag",
    help="Financial RAG retrieval and evaluation system.",
    no_args_is_help=True,
)


@app.callback()
def root() -> None:
    """Financial RAG project commands."""


@app.command()
def info() -> None:
    """Print validated, non-secret application information."""

    settings = get_settings()
    payload = {
        "app": settings.app_name,
        "environment": settings.environment,
        "version": __version__,
    }
    typer.echo(json.dumps(payload, sort_keys=True))


@app.command()
def ingest(
    cik: Annotated[str, typer.Argument(help="Company CIK, with or without leading zeros.")],
    form: Annotated[
        list[str] | None,
        typer.Option("--form", help="Filing form to ingest; repeat for multiple forms."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(min=1, help="Maximum recent filings to ingest."),
    ] = 10,
    user_agent: Annotated[
        str | None,
        typer.Option(help="Descriptive SEC User-Agent; preferably set it in .env."),
    ] = None,
) -> None:
    """Ingest a bounded set of recent SEC 10-K/10-Q filings."""

    settings = get_settings()
    configure_logging(settings.log_level)
    selected_forms = form or ["10-K", "10-Q"]
    invalid_forms = sorted(set(selected_forms) - {"10-K", "10-Q"})
    if invalid_forms:
        joined = ", ".join(invalid_forms)
        raise typer.BadParameter(f"unsupported filing form(s): {joined}", param_hint="--form")
    sec_identity = user_agent or settings.sec_user_agent
    if not sec_identity:
        raise typer.BadParameter(
            "set FINANCIAL_RAG_SEC_USER_AGENT or pass --user-agent",
            param_hint="--user-agent",
        )

    try:
        with SECClient(
            user_agent=sec_identity,
            cache_dir=settings.data_dir / "raw",
            requests_per_second=settings.sec_requests_per_second,
            timeout_seconds=settings.sec_timeout_seconds,
            max_retries=settings.sec_max_retries,
        ) as client:
            pipeline = IngestionPipeline(
                client,
                processed_dir=settings.data_dir / "processed",
                chunk_max_chars=settings.chunk_max_chars,
                chunk_overlap_chars=settings.chunk_overlap_chars,
            )
            stats = pipeline.ingest(
                cik,
                forms=tuple(cast(FilingType, item) for item in selected_forms),
                limit=limit,
            )
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Ingestion failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(stats.model_dump_json())


@app.command("build-index")
def build_index(
    processed_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing processed chunk JSONL files."),
    ] = None,
    index_dir: Annotated[
        Path | None,
        typer.Option(help="Destination directory for the exact dense index."),
    ] = None,
) -> None:
    """Embed all processed chunks and build the dense baseline index."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        stats = DenseIndexBuilder(embedder).build(
            processed_dir=processed_dir or settings.data_dir / "processed",
            index_dir=index_dir or settings.index_dir,
        )
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Index build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(stats.model_dump_json())


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Financial filing question to retrieve for.")],
    top_k: Annotated[int | None, typer.Option(min=1, help="Number of chunks to return.")] = None,
    index_dir: Annotated[
        Path | None,
        typer.Option(help="Dense index directory."),
    ] = None,
) -> None:
    """Search the dense baseline and return ranked evidence chunks."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        response = index.search(
            question,
            embedder,
            top_k=top_k or settings.retrieval_top_k,
        )
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(response.model_dump_json())


@app.command("evaluate-dense")
def evaluate_dense_command(
    dataset: Annotated[
        Path,
        typer.Argument(help="Labeled evaluation dataset in JSONL format."),
    ] = Path("data/eval/dense_baseline.jsonl"),
    top_k: Annotated[int | None, typer.Option(min=1, help="Recall/MRR cutoff.")] = None,
    index_dir: Annotated[
        Path | None,
        typer.Option(help="Dense index directory."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Optional path for the JSON evaluation report."),
    ] = None,
) -> None:
    """Measure Recall@K, MRR, and retrieval latency for the dense baseline."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        report = evaluate_dense(
            index,
            embedder,
            load_evaluation_cases(dataset),
            top_k=top_k or settings.retrieval_top_k,
        )
        if output is not None:
            write_evaluation_report(report, output)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Evaluation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.model_dump_json())


@app.command("query-hybrid")
def query_hybrid(
    question: Annotated[str, typer.Argument(help="Financial filing question to retrieve for.")],
    top_k: Annotated[int | None, typer.Option(min=1, help="Number of chunks to return.")] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Apply the configured cross encoder."),
    ] = False,
    index_dir: Annotated[Path | None, typer.Option(help="Dense index directory.")] = None,
) -> None:
    """Search with dense+BM25 RRF and optional cross-encoder reranking."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        hybrid = _create_hybrid_retriever(settings, index, embedder)
        retriever = (
            RerankingRetriever(
                hybrid,
                _create_reranker(settings),
                candidate_count=settings.rerank_candidates,
            )
            if rerank
            else hybrid
        )
        response = retriever.search(question, top_k=top_k or settings.retrieval_top_k)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(response.model_dump_json())


@app.command("compare-retrieval")
def compare_retrieval(
    dataset: Annotated[
        Path,
        typer.Argument(help="Labeled evaluation dataset in JSONL format."),
    ] = Path("data/eval/dense_baseline.jsonl"),
    top_k: Annotated[int | None, typer.Option(min=1, help="Metric cutoff.")] = None,
    index_dir: Annotated[Path | None, typer.Option(help="Dense index directory.")] = None,
    output: Annotated[
        Path,
        typer.Option(help="Machine-readable comparison report path."),
    ] = Path("experiments/retrieval-comparison.json"),
) -> None:
    """Compare dense, hybrid, reranked, and metadata-routed retrieval."""

    settings = get_settings()
    configure_logging(settings.log_level)
    cutoff = top_k or settings.retrieval_top_k
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        cases = load_evaluation_cases(dataset)
        reports = _evaluate_retrieval_configurations(
            settings,
            index,
            embedder,
            cases,
            cutoff=cutoff,
        )
        comparison = RetrievalComparisonReport(experiments=reports)
        write_comparison_report(comparison, output)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Comparison failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(comparison.model_dump_json())


@app.command("build-fact-index")
def build_fact_index(
    raw_dir: Annotated[
        Path | None,
        typer.Option(help="Directory containing cached inline-XBRL SEC filings."),
    ] = None,
    index_dir: Annotated[Path | None, typer.Option(help="Dense index directory.")] = None,
    output: Annotated[
        Path | None,
        typer.Option(help="Destination JSONL path for normalized numeric facts."),
    ] = None,
) -> None:
    """Extract a local structured-fact index from cached SEC filings."""

    settings = get_settings()
    configure_logging(settings.log_level)
    destination = output or settings.fact_index_path
    try:
        index = DenseIndex.load(index_dir or settings.index_dir)
        store = InlineXBRLStore.extract(raw_dir or settings.data_dir / "raw", index.chunks)
        store.save(destination)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Fact index build failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps({"fact_count": len(store.facts), "path": str(destination)}))


@app.command("ask-financial")
def ask_financial(
    question: Annotated[str, typer.Argument(help="Finance-aware question over local filings.")],
    index_dir: Annotated[Path | None, typer.Option(help="Dense index directory.")] = None,
    fact_index: Annotated[
        Path | None,
        typer.Option(help="Structured XBRL fact index path."),
    ] = None,
) -> None:
    """Route a question to filtered narrative retrieval or structured calculation."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        fact_store = InlineXBRLStore.load(fact_index or settings.fact_index_path)
        engine = FinancialIntelligenceEngine(
            QueryRouter(index.chunks),
            _create_hybrid_retriever(settings, index, embedder),
            fact_store,
            narrative_top_k=settings.retrieval_top_k,
            filter_candidates=max(settings.dense_candidates, settings.bm25_candidates),
        )
        answer = engine.answer(question)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Financial query failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(answer.model_dump_json())


@app.command("run-benchmark")
def run_benchmark(
    retrieval_dataset: Annotated[
        Path,
        typer.Option(help="Labeled retrieval benchmark JSONL."),
    ] = Path("data/eval/retrieval_benchmark.jsonl"),
    answer_dataset: Annotated[
        Path,
        typer.Option(help="Labeled evidence-grounded answer benchmark JSONL."),
    ] = Path("data/eval/answer_benchmark.jsonl"),
    top_k: Annotated[int | None, typer.Option(min=1, help="Retrieval metric cutoff.")] = None,
    index_dir: Annotated[Path | None, typer.Option(help="Dense index directory.")] = None,
    fact_index: Annotated[
        Path | None, typer.Option(help="Structured XBRL fact index path.")
    ] = None,
    output: Annotated[
        Path,
        typer.Option(help="Unified machine-readable benchmark artifact."),
    ] = Path("experiments/phase5-benchmark.json"),
    markdown: Annotated[
        Path,
        typer.Option(help="Rendered comparison and failure report."),
    ] = Path("experiments/phase5-benchmark.md"),
) -> None:
    """Run retrieval, answer, latency, and failure evaluation end to end."""

    settings = get_settings()
    configure_logging(settings.log_level)
    cutoff = top_k or settings.retrieval_top_k
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(index_dir or settings.index_dir)
        retrieval_cases = load_evaluation_cases(retrieval_dataset)
        answer_cases = load_answer_cases(answer_dataset)
        retrieval_reports = _evaluate_retrieval_configurations(
            settings,
            index,
            embedder,
            retrieval_cases,
            cutoff=cutoff,
        )
        fact_store = InlineXBRLStore.load(fact_index or settings.fact_index_path)
        engine = FinancialIntelligenceEngine(
            QueryRouter(index.chunks),
            _create_hybrid_retriever(settings, index, embedder),
            fact_store,
            narrative_top_k=cutoff,
            filter_candidates=max(settings.dense_candidates, settings.bm25_candidates),
        )
        answer_report, answer_failures = evaluate_answers(
            engine,
            answer_cases,
            corpus=index.chunks,
            facts=fact_store.facts,
        )
        report = build_benchmark_report(
            retrieval_reports,
            answer_report,
            answer_failures,
        )
        write_benchmark_artifacts(report, output, markdown)
    except (FinancialRAGError, ValueError) as exc:
        typer.echo(f"Benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "summary": report.summary.model_dump(mode="json"),
                "json_report": str(output),
                "markdown_report": str(markdown),
            },
            sort_keys=True,
        )
    )


@app.command()
def serve(
    host: Annotated[str | None, typer.Option(help="HTTP bind host.")] = None,
    port: Annotated[int | None, typer.Option(min=1, max=65535, help="HTTP bind port.")] = None,
) -> None:
    """Serve the production FastAPI application with one local-model worker."""

    import uvicorn

    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "financial_rag.api:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        workers=1,
        log_config=None,
    )


@app.command("run-final-benchmark")
def run_final_benchmark(
    retrieval_dataset: Annotated[
        Path,
        typer.Option(help="Ground-truth retrieval benchmark JSONL."),
    ] = Path("data/eval/retrieval_benchmark.jsonl"),
    answer_dataset: Annotated[
        Path,
        typer.Option(help="Ground-truth financial answer benchmark JSONL."),
    ] = Path("data/eval/answer_benchmark.jsonl"),
    experiments_dir: Annotated[
        Path,
        typer.Option(help="Directory for final machine-readable artifacts."),
    ] = Path("experiments"),
    benchmark_doc: Annotated[
        Path,
        typer.Option(help="Generated human-readable benchmark document."),
    ] = Path("docs/BENCHMARK.md"),
    resume_doc: Annotated[
        Path,
        typer.Option(help="Generated, measured resume bullet document."),
    ] = Path("docs/RESUME_METRICS.md"),
    measure_indexing: Annotated[
        bool,
        typer.Option(
            "--measure-indexing/--skip-indexing",
            help="Rebuild into a temporary directory to measure indexing time.",
        ),
    ] = True,
    indexing_time_ms: Annotated[
        float | None,
        typer.Option(
            min=0,
            help="Previously measured full index build time for this exact corpus.",
        ),
    ] = None,
) -> None:
    """Run the canonical multi-cutoff audit and generate final portfolio artifacts."""

    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        embedder = _create_embedder(settings)
        index = DenseIndex.load(settings.index_dir)
        retrieval_cases = load_evaluation_cases(retrieval_dataset)
        answer_cases = load_answer_cases(answer_dataset)
        fact_store = InlineXBRLStore.load(settings.fact_index_path)
        dense = DenseRetriever(index, embedder)
        hybrid = _create_hybrid_retriever(settings, index, embedder)
        reranker = _create_reranker(settings)
        rerank_candidates = max(settings.rerank_candidates, 10)
        reranked = RerankingRetriever(
            hybrid,
            reranker,
            candidate_count=rerank_candidates,
        )
        routed = RoutedRetriever(
            hybrid,
            QueryRouter(index.chunks),
            candidate_count=settings.dense_candidates + settings.bm25_candidates,
        )
        dense_report = evaluate_final_retriever(
            experiment_name="Dense",
            configuration={"embedding_model": embedder.model_name, "top_k": 10},
            retriever=dense,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=retrieval_cases,
            top_k=10,
        )
        hybrid_report = evaluate_final_retriever(
            experiment_name="Hybrid",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "reranker": False,
                "top_k": 10,
            },
            retriever=hybrid,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=retrieval_cases,
            top_k=10,
        )
        reranked_report = evaluate_final_retriever(
            experiment_name="Hybrid + Reranker",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "reranker_model": reranker.model_name,
                "rerank_candidates": rerank_candidates,
                "top_k": 10,
            },
            retriever=reranked,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=retrieval_cases,
            top_k=10,
        )
        reranked_report = classify_reranker_drops(hybrid_report, reranked_report)
        final_report = evaluate_final_retriever(
            experiment_name="Final (Routed Hybrid)",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "metadata_filters": True,
                "temporal_filters": True,
                "reranker": False,
                "top_k": 10,
            },
            retriever=routed,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=retrieval_cases,
            top_k=10,
        )
        engine = FinancialIntelligenceEngine(
            QueryRouter(index.chunks),
            hybrid,
            fact_store,
            narrative_top_k=settings.retrieval_top_k,
            filter_candidates=settings.dense_candidates + settings.bm25_candidates,
        )
        answer_report, _ = evaluate_answers(
            engine,
            answer_cases,
            corpus=index.chunks,
            facts=fact_store.facts,
        )
        corpus = measure_corpus(
            chunks=index.chunks,
            fact_count=len(fact_store.facts),
            raw_dir=settings.data_dir / "raw",
            processed_dir=settings.data_dir / "processed",
            index_dir=settings.index_dir,
            vector_count=index.manifest.document_count,
            vector_dimension=index.manifest.dimension,
            retrieval_questions=len(retrieval_cases),
            answer_questions=len(answer_cases),
        )
        measured_indexing_time = (
            indexing_time_ms
            if indexing_time_ms is not None
            else (_measure_indexing(settings, embedder) if measure_indexing else None)
        )
        report = build_final_report(
            corpus=corpus,
            dense=dense_report,
            hybrid=hybrid_report,
            hybrid_reranker=reranked_report,
            final_pipeline=final_report,
            answer_report=answer_report,
            answer_cases=answer_cases,
            retrieval_cases=retrieval_cases,
            indexing_time_ms=measured_indexing_time,
        )
        write_final_artifacts(report, experiments_dir)
        write_final_documents(report, benchmark_doc, resume_doc)
        verify_final_artifact(
            experiments_dir / "final_benchmark.json",
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            retrieval_cases=retrieval_cases,
            answer_cases=answer_cases,
        )
    except (FinancialRAGError, OSError, ValueError) as exc:
        typer.echo(f"Final benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "artifact": str(experiments_dir / "final_benchmark.json"),
                "dense": report.dense.metrics.model_dump(mode="json"),
                "final": report.final_pipeline.metrics.model_dump(mode="json"),
                "improvements": report.improvements.model_dump(mode="json"),
            },
            sort_keys=True,
        )
    )


@app.command("freeze-benchmark-v2")
def freeze_benchmark_v2_command(
    retrieval_output: Annotated[Path, typer.Option(help="Frozen retrieval JSONL output.")] = Path(
        "data/eval/benchmark_v2_retrieval.jsonl"
    ),
    answer_output: Annotated[
        Path, typer.Option(help="Frozen numerical answer JSONL output.")
    ] = Path("data/eval/benchmark_v2_answers.jsonl"),
    manifest_output: Annotated[
        Path, typer.Option(help="Frozen benchmark integrity manifest.")
    ] = Path("experiments/benchmark_v2/manifest.json"),
) -> None:
    """Freeze Benchmark V2 from SEC provenance and XBRL source facts."""

    settings = get_settings()
    try:
        manifest = freeze_benchmark_v2(
            settings.data_dir / "processed",
            settings.fact_index_path,
            retrieval_output,
            answer_output,
            manifest_output,
        )
    except (FinancialRAGError, OSError, ValueError) as exc:
        typer.echo(f"Benchmark V2 freeze failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(manifest, sort_keys=True))


@app.command("reproduce-benchmark-v2")
def reproduce_benchmark_v2(
    manifest_path: Annotated[
        Path,
        typer.Option(help="Checked-in exact SEC corpus manifest."),
    ] = Path("experiments/benchmark_v2/corpus_manifest.json"),
    user_agent: Annotated[
        str | None,
        typer.Option(help="Descriptive SEC User-Agent; preferably set it in .env."),
    ] = None,
    rebuild_index: Annotated[
        bool,
        typer.Option(
            "--rebuild-index/--reuse-index",
            help="Build the dense index, or validate an existing local index.",
        ),
    ] = True,
) -> None:
    """Rebuild the exact public SEC corpus and indexes for frozen Benchmark V2."""

    settings = get_settings()
    configure_logging(settings.log_level)
    sec_identity = user_agent or settings.sec_user_agent
    if not sec_identity:
        raise typer.BadParameter(
            "set FINANCIAL_RAG_SEC_USER_AGENT or pass --user-agent",
            param_hint="--user-agent",
        )
    try:
        manifest = load_corpus_manifest(manifest_path)
        if settings.embedding_model != manifest.embedding_model:
            raise ReproductionError("configured embedding model does not match corpus manifest")
        if (
            settings.chunk_max_chars != manifest.chunk_max_chars
            or settings.chunk_overlap_chars != manifest.chunk_overlap_chars
        ):
            raise ReproductionError("configured chunk sizes do not match corpus manifest")
        verify_frozen_datasets(manifest, Path.cwd())
        with SECClient(
            user_agent=sec_identity,
            cache_dir=settings.data_dir / "raw",
            requests_per_second=settings.sec_requests_per_second,
            timeout_seconds=settings.sec_timeout_seconds,
            max_retries=settings.sec_max_retries,
        ) as client:
            ingestion = prepare_benchmark_corpus(
                manifest,
                client,
                processed_dir=settings.data_dir / "processed",
            )
        if rebuild_index:
            embedder = _create_embedder(settings)
            index_stats = DenseIndexBuilder(embedder).build(
                processed_dir=settings.data_dir / "processed",
                index_dir=settings.index_dir,
            )
            corpus_fingerprint = index_stats.corpus_fingerprint
        else:
            corpus_fingerprint = DenseIndex.load(settings.index_dir).manifest.corpus_fingerprint
        if corpus_fingerprint != manifest.expected_corpus_fingerprint:
            raise ReproductionError("dense corpus fingerprint does not match manifest")
        index = DenseIndex.load(settings.index_dir)
        fact_store = InlineXBRLStore.extract(settings.data_dir / "raw", index.chunks)
        fact_store.save(settings.fact_index_path)
        if len(fact_store.facts) != manifest.expected_xbrl_fact_count:
            raise ReproductionError("XBRL fact count does not match manifest")
        if text_sha256(settings.fact_index_path) != manifest.expected_fact_index_sha256:
            raise ReproductionError("XBRL fact index hash does not match manifest")
        if file_sha256(Path.cwd() / manifest.retrieval_dataset) != manifest.retrieval_sha256:
            raise ReproductionError("retrieval ground truth changed during reproduction")
        if file_sha256(Path.cwd() / manifest.answer_dataset) != manifest.answer_sha256:
            raise ReproductionError("answer ground truth changed during reproduction")
    except (FinancialRAGError, OSError, ValueError) as exc:
        typer.echo(f"Benchmark V2 reproduction failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "benchmark_version": manifest.benchmark_version,
                "chunks": ingestion.chunks_written,
                "corpus_fingerprint": corpus_fingerprint,
                "filings": ingestion.processed_documents,
                "status": "prepared",
                "xbrl_facts": len(fact_store.facts),
            },
            sort_keys=True,
        )
    )


@app.command("verify-final-benchmark")
def verify_final_benchmark(
    artifact: Annotated[
        Path,
        typer.Option(help="Canonical final benchmark JSON artifact."),
    ] = Path("experiments/final_benchmark.json"),
    retrieval_dataset: Annotated[
        Path,
        typer.Option(help="Ground-truth retrieval benchmark JSONL."),
    ] = Path("data/eval/retrieval_benchmark.jsonl"),
    answer_dataset: Annotated[
        Path,
        typer.Option(help="Ground-truth financial answer benchmark JSONL."),
    ] = Path("data/eval/answer_benchmark.jsonl"),
) -> None:
    """Recompute stored metrics and verify corpus/dataset fingerprints."""

    settings = get_settings()
    try:
        index = DenseIndex.load(settings.index_dir)
        report = verify_final_artifact(
            artifact,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            retrieval_cases=load_evaluation_cases(retrieval_dataset),
            answer_cases=load_answer_cases(answer_dataset),
        )
    except (FinancialRAGError, OSError, ValueError) as exc:
        typer.echo(f"Final benchmark verification failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        json.dumps(
            {
                "artifact": str(artifact),
                "generated_at": report.generated_at.isoformat(),
                "status": "verified",
            },
            sort_keys=True,
        )
    )


def _create_embedder(settings: Settings) -> FastEmbedModel:
    return FastEmbedModel(
        settings.embedding_model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.embedding_batch_size,
    )


def _create_reranker(settings: Settings) -> FastEmbedReranker:
    return FastEmbedReranker(
        settings.reranker_model,
        cache_dir=settings.model_cache_dir,
        batch_size=settings.reranker_batch_size,
    )


def _create_hybrid_retriever(
    settings: Settings,
    index: DenseIndex,
    embedder: FastEmbedModel,
) -> HybridRetriever:
    dense = DenseRetriever(index, embedder)
    bm25 = BM25Index(index.chunks, k1=settings.bm25_k1, b=settings.bm25_b)
    return HybridRetriever(
        dense,
        bm25,
        dense_candidates=settings.dense_candidates,
        bm25_candidates=settings.bm25_candidates,
        rrf_k=settings.rrf_k,
    )


def _evaluate_retrieval_configurations(
    settings: Settings,
    index: DenseIndex,
    embedder: FastEmbedModel,
    cases: list[EvaluationCase],
    *,
    cutoff: int,
) -> tuple[RetrievalEvaluationReport, ...]:
    dense = DenseRetriever(index, embedder)
    hybrid = _create_hybrid_retriever(settings, index, embedder)
    reranker = _create_reranker(settings)
    reranked = RerankingRetriever(
        hybrid,
        reranker,
        candidate_count=settings.rerank_candidates,
    )
    routed = RoutedRetriever(
        hybrid,
        QueryRouter(index.chunks),
        candidate_count=settings.dense_candidates + settings.bm25_candidates,
    )
    return (
        evaluate_retriever(
            experiment_name="E1 Dense",
            configuration={"top_k": cutoff},
            model_name=embedder.model_name,
            retriever=dense,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=cases,
            top_k=cutoff,
        ),
        evaluate_retriever(
            experiment_name="E2 Hybrid RRF",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "top_k": cutoff,
            },
            model_name=embedder.model_name,
            retriever=hybrid,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=cases,
            top_k=cutoff,
        ),
        evaluate_retriever(
            experiment_name="E3 Hybrid RRF + Reranker",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "rerank_candidates": settings.rerank_candidates,
                "top_k": cutoff,
            },
            model_name=f"{embedder.model_name} + {reranker.model_name}",
            retriever=reranked,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=cases,
            top_k=cutoff,
        ),
        evaluate_retriever(
            experiment_name="E4 Routed Hybrid RRF",
            configuration={
                "dense_candidates": settings.dense_candidates,
                "bm25_candidates": settings.bm25_candidates,
                "rrf_k": settings.rrf_k,
                "route_filters": "company,filing_type,report_year",
                "top_k": cutoff,
            },
            model_name=embedder.model_name,
            retriever=routed,
            corpus_fingerprint=index.manifest.corpus_fingerprint,
            corpus=index.chunks,
            cases=cases,
            top_k=cutoff,
        ),
    )


def _measure_indexing(settings: Settings, embedder: FastEmbedModel) -> float:
    parent = settings.data_dir / "index"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="audit-index-", dir=parent) as temporary:
        stats = DenseIndexBuilder(embedder).build(
            processed_dir=settings.data_dir / "processed",
            index_dir=Path(temporary),
        )
    return stats.indexing_latency_ms


def main() -> None:
    """Run the command-line application."""

    app()
