"""Local inline-XBRL extraction, persistence, and structured lookup."""

import re
from collections.abc import Iterable
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from bs4 import BeautifulSoup, Tag
from pydantic import ValidationError

from financial_rag.errors import FinancialIntelligenceError
from financial_rag.ingestion.models import DocumentChunk, FilingType
from financial_rag.intelligence.models import XBRLFact

_NUMERIC_CLEANUP = re.compile(r"[^0-9.\-]")


class InlineXBRLStore:
    """An immutable local collection of normalized SEC inline-XBRL facts."""

    def __init__(self, facts: Iterable[XBRLFact]) -> None:
        self._facts = tuple(facts)

    @property
    def facts(self) -> tuple[XBRLFact, ...]:
        return self._facts

    @classmethod
    def extract(cls, raw_dir: Path, chunks: Iterable[DocumentChunk]) -> "InlineXBRLStore":
        """Extract numeric facts from cached filings represented in the dense corpus."""

        filings = {}
        for chunk in chunks:
            filings[chunk.metadata.accession_number.replace("-", "")] = chunk.metadata
        facts: list[XBRLFact] = []
        for path in sorted(raw_dir.rglob("*.htm*")):
            metadata = filings.get(path.parent.name)
            if metadata is None:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise FinancialIntelligenceError(f"failed to read cached filing {path}") from exc
            soup = BeautifulSoup(content, "html.parser")
            contexts = {
                str(context.get("id")): context
                for context in soup.find_all("xbrli:context")
                if context.get("id")
            }
            for element in soup.find_all("ix:nonfraction"):
                fact = _parse_fact(element, contexts, metadata)
                if fact is not None:
                    facts.append(fact)
        if not facts:
            raise FinancialIntelligenceError(f"no inline-XBRL numeric facts found under {raw_dir}")
        return cls(facts)

    @classmethod
    def load(cls, path: Path) -> "InlineXBRLStore":
        """Load and validate a JSONL fact index."""

        try:
            facts = [
                XBRLFact.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValidationError) as exc:
            raise FinancialIntelligenceError(f"failed to load XBRL fact index from {path}") from exc
        if not facts:
            raise FinancialIntelligenceError(f"XBRL fact index is empty: {path}")
        return cls(facts)

    def save(self, path: Path) -> None:
        """Persist facts atomically as stable JSONL records."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        try:
            temporary.write_text(
                "".join(f"{fact.model_dump_json()}\n" for fact in self._facts),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            raise FinancialIntelligenceError(f"failed to write XBRL fact index to {path}") from exc

    def best_fact(
        self,
        concepts: tuple[str, ...],
        *,
        cik: str | None,
        year: int | None,
        filing_types: tuple[FilingType, ...] = (),
    ) -> XBRLFact | None:
        """Select the most specific consolidated fact for a company and period."""

        concept_priority = {concept.casefold(): index for index, concept in enumerate(concepts)}
        candidates = [
            fact
            for fact in self._facts
            if fact.concept.casefold() in concept_priority
            and (cik is None or fact.cik == cik)
            and (year is None or fact.period_end.year == year)
            and (not filing_types or fact.filing_type in filing_types)
        ]
        if not candidates:
            return None

        def rank(fact: XBRLFact) -> tuple[int, int, int, int, int, int]:
            duration = (fact.period_end - fact.period_start).days if fact.period_start else 0
            annual_preference = 0 if year is not None and duration >= 300 else 1
            report_gap = (
                abs((fact.report_date - fact.period_end).days)
                if year is not None and fact.report_date is not None
                else 0
            )
            return (
                int(fact.has_dimensions),
                concept_priority[fact.concept.casefold()],
                annual_preference,
                report_gap,
                -fact.filing_date.toordinal(),
                -duration,
            )

        return min(candidates, key=rank)


def _parse_fact(element: Tag, contexts: dict[str, Tag], metadata: object) -> XBRLFact | None:
    from financial_rag.ingestion.models import ChunkMetadata

    if not isinstance(metadata, ChunkMetadata):
        return None
    concept = _attribute(element, "name")
    context_ref = _attribute(element, "contextref")
    unit = _attribute(element, "unitref")
    if not concept or not context_ref or not unit:
        return None
    context = contexts.get(context_ref)
    if context is None:
        return None
    period_start = _date_child(context, "xbrli:startdate")
    period_end = _date_child(context, "xbrli:enddate") or _date_child(context, "xbrli:instant")
    if period_end is None:
        return None
    value = _numeric_value(element)
    if value is None:
        return None
    return XBRLFact(
        concept=concept,
        value=value,
        unit=unit,
        period_start=period_start,
        period_end=period_end,
        has_dimensions=context.find(["xbrli:segment", "xbrli:scenario", "xbrldi:explicitmember"])
        is not None,
        company=metadata.company,
        ticker=metadata.ticker,
        cik=metadata.cik,
        filing_type=metadata.filing_type,
        filing_date=metadata.filing_date,
        report_date=metadata.report_date,
        accession_number=metadata.accession_number,
        source_url=metadata.source_url,
    )


def _attribute(element: Tag, name: str) -> str | None:
    value = element.get(name)
    return value if isinstance(value, str) else None


def _date_child(context: Tag, name: str) -> date | None:
    child = context.find(name)
    if not isinstance(child, Tag):
        return None
    try:
        return date.fromisoformat(child.get_text(strip=True))
    except ValueError:
        return None


def _numeric_value(element: Tag) -> Decimal | None:
    raw = element.get_text("", strip=True).replace("\u2212", "-").replace("\u2014", "")
    negative_parentheses = raw.startswith("(") and raw.endswith(")")
    normalized = _NUMERIC_CLEANUP.sub("", raw)
    if normalized in {"", "-", "."}:
        return None
    try:
        value = Decimal(normalized)
        scale = int(_attribute(element, "scale") or "0")
    except (InvalidOperation, ValueError):
        return None
    if negative_parentheses or _attribute(element, "sign") == "-":
        value = -abs(value)
    return value * (Decimal(10) ** scale)
