"""Deterministic construction and freezing of the expanded SEC benchmark."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from financial_rag.evaluation.models import AnswerEvaluationCase, EvaluationCase
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.intelligence.models import XBRLFact
from financial_rag.intelligence.routing import METRIC_CONCEPTS
from financial_rag.intelligence.xbrl import InlineXBRLStore

BENCHMARK_VERSION = "benchmark_v2"
RETRIEVAL_QUESTION_COUNT = 150

_SECTION_KINDS = {
    "discussion": ("item 7", "item 2"),
    "risks": ("item 1a",),
    "financial_statements": ("item 8", "item 1"),
}
_CATEGORIES = (
    "narrative_semantic",
    "exact_financial_terminology",
    "company_specific",
    "filing_specific",
    "temporal",
    "cross_period",
    "numerical_xbrl",
    "similar_information_across_filings",
)


def freeze_benchmark_v2(
    processed_dir: Path,
    fact_index: Path,
    retrieval_output: Path,
    answer_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """Create a fixed benchmark from provenance and SEC evidence, never retrieval output."""

    chunks = _load_chunks(processed_dir)
    retrieval_cases = _retrieval_cases(chunks)
    answer_cases = _answer_cases(InlineXBRLStore.load(fact_index))
    corpus_summary = _validate_ground_truth(chunks, retrieval_cases, answer_cases)
    _write_jsonl(retrieval_output, retrieval_cases)
    _write_jsonl(answer_output, answer_cases)
    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "frozen": True,
        "construction": (
            "SEC filing provenance, section metadata, and inline XBRL facts; no retrieval outputs"
        ),
        "retrieval_questions": len(retrieval_cases),
        "answer_questions": len(answer_cases),
        "category_distribution": dict(
            sorted(Counter(case.category for case in retrieval_cases).items())
        ),
        **corpus_summary,
        "integrity_checks": "passed",
        "retrieval_sha256": _sha256(retrieval_output),
        "answer_sha256": _sha256(answer_output),
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _validate_ground_truth(
    chunks: tuple[DocumentChunk, ...],
    retrieval_cases: list[EvaluationCase],
    answer_cases: list[AnswerEvaluationCase],
) -> dict[str, int]:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    filings = {chunk.metadata.accession_number for chunk in chunks}
    companies = {chunk.metadata.cik for chunk in chunks}
    if len(companies) < 5 or len(filings) < 50:
        raise ValueError("Benchmark V2 requires at least 5 companies and 50 filings")
    if len(retrieval_cases) != RETRIEVAL_QUESTION_COUNT:
        raise ValueError("Benchmark V2 requires exactly 150 retrieval questions")
    for case in retrieval_cases:
        if not case.relevant_chunk_ids or not case.expected_evidence or not case.expected_source:
            raise ValueError(f"incomplete retrieval ground truth: {case.question}")
        evidence_chunks = [chunks_by_id.get(chunk_id) for chunk_id in case.relevant_chunk_ids]
        if any(chunk is None for chunk in evidence_chunks):
            raise ValueError(f"unknown relevant chunk ID: {case.question}")
        if not all(
            chunk is not None
            and chunk.metadata.company == case.expected_company
            and chunk.metadata.accession_number == case.expected_accession_number
            and chunk.metadata.source_url == case.expected_source
            for chunk in evidence_chunks
        ):
            raise ValueError(f"retrieval provenance mismatch: {case.question}")
    if len(answer_cases) < 20:
        raise ValueError("Benchmark V2 requires at least 20 labeled numerical cases")
    if any(
        not case.expected_values
        or not case.expected_concept
        or not case.expected_accession_number
        or not case.expected_source
        for case in answer_cases
    ):
        raise ValueError("incomplete numerical ground truth")
    return {
        "company_count": len(companies),
        "filing_count": len(filings),
        "chunk_count": len(chunks),
    }


def _load_chunks(processed_dir: Path) -> tuple[DocumentChunk, ...]:
    chunks: list[DocumentChunk] = []
    for path in sorted(processed_dir.rglob("*.jsonl")):
        chunks.extend(
            DocumentChunk.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not chunks:
        raise ValueError("processed corpus is empty")
    return tuple(chunks)


def _retrieval_cases(chunks: tuple[DocumentChunk, ...]) -> list[EvaluationCase]:
    filings: dict[str, list[DocumentChunk]] = defaultdict(list)
    for chunk in chunks:
        filings[chunk.metadata.accession_number].append(chunk)
    candidates: list[EvaluationCase] = []
    ordered_filings = sorted(
        filings.values(),
        key=lambda items: (
            items[0].metadata.cik,
            items[0].metadata.report_date or items[0].metadata.filing_date,
            items[0].metadata.accession_number,
        ),
    )
    for filing_index, filing_chunks in enumerate(ordered_filings):
        for kind_index, kind in enumerate(_SECTION_KINDS):
            evidence_chunks = _section_chunks(filing_chunks, kind)
            if not evidence_chunks:
                continue
            metadata = evidence_chunks[0].metadata
            category = _CATEGORIES[(filing_index * 3 + kind_index) % len(_CATEGORIES)]
            period = metadata.report_date or metadata.filing_date
            question = _question(
                category, kind, metadata.company, metadata.filing_type, period.isoformat()
            )
            evidence_chunk = max(evidence_chunks, key=lambda chunk: len(chunk.text))
            evidence = re.sub(r"\s+", " ", evidence_chunk.text).strip()[:500]
            candidates.append(
                EvaluationCase(
                    question=question,
                    relevant_chunk_ids=frozenset(chunk.chunk_id for chunk in evidence_chunks),
                    expected_company=metadata.company,
                    expected_filing=metadata.filing_type,
                    expected_period=metadata.report_date,
                    expected_section=evidence_chunks[0].metadata.section,
                    expected_accession_number=metadata.accession_number,
                    benchmark_version=BENCHMARK_VERSION,
                    category=category,
                    expected_source=metadata.source_url,
                    expected_evidence=evidence,
                )
            )
    if len(candidates) < RETRIEVAL_QUESTION_COUNT:
        raise ValueError(f"only {len(candidates)} grounded retrieval cases could be built")
    # Round-robin categories prevents the first 150 filings/sections from skewing the mix.
    by_category: dict[str, list[EvaluationCase]] = defaultdict(list)
    for case in candidates:
        by_category[case.category or "unknown"].append(case)
    selected: list[EvaluationCase] = []
    while len(selected) < RETRIEVAL_QUESTION_COUNT:
        progressed = False
        for category in _CATEGORIES:
            if by_category[category] and len(selected) < RETRIEVAL_QUESTION_COUNT:
                selected.append(by_category[category].pop(0))
                progressed = True
        if not progressed:
            break
    if len({case.question for case in selected}) != len(selected):
        raise ValueError("benchmark contains duplicate questions")
    return selected


def _section_chunks(chunks: list[DocumentChunk], kind: str) -> list[DocumentChunk]:
    prefixes = _SECTION_KINDS[kind]
    form = chunks[0].metadata.filing_type
    preferred = prefixes[0] if form == "10-K" else prefixes[-1]
    item_number = preferred.removeprefix("item ")
    pattern = re.compile(rf"^item\s*{re.escape(item_number)}(?:[.\s]|$)", re.IGNORECASE)
    return [chunk for chunk in chunks if pattern.match(chunk.metadata.section)]


def _question(category: str, kind: str, company: str, form: str, period: str) -> str:
    topic = {
        "discussion": "management's discussion of financial condition and results of operations",
        "risks": "risk factors",
        "financial_statements": "financial statements and related disclosures",
    }[kind]
    if category == "exact_financial_terminology":
        return f"What does {company}'s {form} for the period ended {period} disclose under {topic}?"
    if category == "company_specific":
        return f"What company-specific information did {company} report about {topic} for {period}?"
    if category == "filing_specific":
        return (
            f"In accession-specific {form} evidence for {company} dated {period}, "
            f"what is reported about {topic}?"
        )
    if category == "temporal":
        return (
            f"What did {company} report about {topic} specifically for the period ended {period}?"
        )
    if category == "cross_period":
        return (
            f"For {company}, what {topic} evidence belongs to {period} rather than "
            "adjacent reporting periods?"
        )
    if category == "numerical_xbrl":
        return (
            f"Which reported figures and financial evidence appear in {company}'s "
            f"{topic} for {period}?"
        )
    if category == "similar_information_across_filings":
        return (
            f"Among similar {company} filings, which {topic} disclosure is from the "
            f"{form} ending {period}?"
        )
    return f"How did {company} describe {topic} in its {form} for the period ended {period}?"


def _answer_cases(store: InlineXBRLStore) -> list[AnswerEvaluationCase]:
    annual_facts = [fact for fact in store.facts if fact.filing_type == "10-K"]
    companies = sorted({(fact.cik, fact.company) for fact in annual_facts})
    cases: list[AnswerEvaluationCase] = []
    metric_labels = (
        ("revenue", "revenue"),
        ("net income", "net income"),
        ("assets", "assets"),
        ("operating cash flow", "operating cash flow"),
        ("research and development", "research and development expense"),
    )
    for cik, company in companies:
        company_facts = [fact for fact in annual_facts if fact.cik == cik]
        latest_year = max((fact.report_date or fact.period_end).year for fact in company_facts)
        for metric, label in metric_labels:
            fact = _best_source_fact(company_facts, METRIC_CONCEPTS[metric], latest_year)
            if fact is None:
                continue
            cases.append(
                AnswerEvaluationCase(
                    question=f"What was {company} {label} in {latest_year} annual filing?",
                    expected_status="answered",
                    expected_route="structured_fact",
                    expected_values={str(latest_year): fact.value},
                    minimum_citations=1,
                    expected_company=company,
                    expected_period_years=(latest_year,),
                    benchmark_version=BENCHMARK_VERSION,
                    expected_concept=fact.concept,
                    expected_accession_number=fact.accession_number,
                    expected_source=fact.source_url,
                )
            )
    return cases


def _best_source_fact(
    facts: list[XBRLFact], concepts: tuple[str, ...], year: int
) -> XBRLFact | None:
    priorities = {concept.casefold(): index for index, concept in enumerate(concepts)}
    candidates = [
        fact
        for fact in facts
        if fact.concept.casefold() in priorities
        and fact.period_end.year == year
        and not fact.has_dimensions
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda fact: (
            priorities[fact.concept.casefold()],
            abs(((fact.report_date or fact.filing_date) - fact.period_end).days),
            -((fact.period_end - fact.period_start).days if fact.period_start else 0),
        ),
    )


def _write_jsonl(path: Path, models: list[EvaluationCase] | list[AnswerEvaluationCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for model in models:
        payload = model.model_dump(mode="json")
        if isinstance(model, EvaluationCase):
            payload["relevant_chunk_ids"] = sorted(model.relevant_chunk_ids)
        lines.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
