"""Finance-aware orchestration with evidence-first answer behavior."""

from decimal import Decimal

from financial_rag.errors import FinancialIntelligenceError
from financial_rag.intelligence.calculations import difference, percent_change, ratio
from financial_rag.intelligence.filters import FilteredRetriever
from financial_rag.intelligence.models import (
    EvidenceCitation,
    FinancialAnswer,
    QueryPlan,
    XBRLFact,
)
from financial_rag.intelligence.routing import METRIC_CONCEPTS, QueryRouter
from financial_rag.intelligence.xbrl import InlineXBRLStore
from financial_rag.retrieval.base import Retriever


class FinancialIntelligenceEngine:
    """Route questions to narrative retrieval or deterministic structured lookup."""

    def __init__(
        self,
        router: QueryRouter,
        retriever: Retriever,
        facts: InlineXBRLStore,
        *,
        narrative_top_k: int = 5,
        filter_candidates: int = 30,
    ) -> None:
        self._router = router
        self._retriever = retriever
        self._facts = facts
        self._narrative_top_k = narrative_top_k
        self._filter_candidates = filter_candidates

    def answer(self, question: str) -> FinancialAnswer:
        """Return only answers supported by cited local filing evidence."""

        plan = self._router.plan(question)
        if plan.intent == "narrative":
            return self._narrative(plan)
        if plan.intent == "calculation":
            return self._calculation(plan)
        return self._structured(plan)

    def _narrative(self, plan: QueryPlan) -> FinancialAnswer:
        response = FilteredRetriever(
            self._retriever,
            plan.filters,
            candidate_count=self._filter_candidates,
        ).search(plan.query, top_k=self._narrative_top_k)
        if not response.results:
            return _insufficient(
                plan,
                "No filing passages matched the requested company and period.",
            )
        citations = tuple(
            EvidenceCitation(
                source_url=result.chunk.metadata.source_url,
                accession_number=result.chunk.metadata.accession_number,
                locator=f"section={result.chunk.metadata.section};chunk={result.chunk.chunk_id}",
                excerpt=result.chunk.text[:300],
            )
            for result in response.results
        )
        return FinancialAnswer(
            question=plan.query,
            status="answered",
            route=plan.intent,
            answer=(
                f"Retrieved {len(citations)} evidence passages; "
                "no unsupported synthesis was generated."
            ),
            citations=citations,
            plan=plan,
        )

    def _structured(self, plan: QueryPlan) -> FinancialAnswer:
        years: tuple[int | None, ...] = plan.years or (None,)
        found: list[tuple[int | None, XBRLFact]] = []
        for year in years:
            fact = self._facts.best_fact(
                plan.concepts,
                cik=plan.filters.cik,
                year=year,
                filing_types=plan.filters.filing_types,
            )
            if fact is None:
                return _insufficient(
                    plan,
                    f"No matching XBRL fact was found for period {year or 'latest'}.",
                )
            found.append((year, fact))
        values = {str(year or fact.period_end.year): fact.value for year, fact in found}
        rendered = "; ".join(
            f"{plan.metric} {year or fact.period_end.year}: "
            f"{_format_decimal(fact.value)} {fact.unit}"
            for year, fact in found
        )
        return FinancialAnswer(
            question=plan.query,
            status="answered",
            route=plan.intent,
            answer=rendered,
            values=values,
            citations=tuple(_fact_citation(fact) for _, fact in found),
            plan=plan,
        )

    def _calculation(self, plan: QueryPlan) -> FinancialAnswer:
        if plan.calculation == "gross_margin":
            if len(plan.years) != 1:
                return _insufficient(plan, "Gross margin requires exactly one explicit year.")
            year = plan.years[0]
            gross_profit = self._lookup(plan, METRIC_CONCEPTS["gross profit"], year)
            revenue = self._lookup(plan, METRIC_CONCEPTS["revenue"], year)
            if gross_profit is None or revenue is None:
                return _insufficient(plan, "Gross profit and revenue evidence are both required.")
            try:
                result = ratio(gross_profit.value, revenue.value)
            except FinancialIntelligenceError as exc:
                return _insufficient(plan, str(exc))
            return FinancialAnswer(
                question=plan.query,
                status="answered",
                route=plan.intent,
                answer=f"Gross margin for {year}: {_format_percentage(result)}%.",
                values={
                    "gross_profit": gross_profit.value,
                    "revenue": revenue.value,
                    "result": result,
                },
                citations=(_fact_citation(gross_profit), _fact_citation(revenue)),
                plan=plan,
            )

        if len(plan.years) != 2:
            return _insufficient(plan, "A change calculation requires exactly two explicit years.")
        prior_year, current_year = sorted(plan.years)
        prior = self._lookup(plan, plan.concepts, prior_year)
        current = self._lookup(plan, plan.concepts, current_year)
        if prior is None or current is None:
            return _insufficient(plan, "Both requested periods require matching XBRL evidence.")
        try:
            result = (
                percent_change(current.value, prior.value)
                if plan.calculation == "percent_change"
                else difference(current.value, prior.value)
            )
        except FinancialIntelligenceError as exc:
            return _insufficient(plan, str(exc))
        suffix = "%" if plan.calculation == "percent_change" else f" {current.unit}"
        return FinancialAnswer(
            question=plan.query,
            status="answered",
            route=plan.intent,
            answer=(
                f"{plan.metric} {plan.calculation} from {prior_year} to {current_year}: "
                f"{_format_percentage(result) if suffix == '%' else _format_decimal(result)}"
                f"{suffix}."
            ),
            values={"prior": prior.value, "current": current.value, "result": result},
            citations=(_fact_citation(prior), _fact_citation(current)),
            plan=plan,
        )

    def _lookup(self, plan: QueryPlan, concepts: tuple[str, ...], year: int) -> XBRLFact | None:
        return self._facts.best_fact(
            concepts,
            cik=plan.filters.cik,
            year=year,
            filing_types=plan.filters.filing_types,
        )


def _fact_citation(fact: XBRLFact) -> EvidenceCitation:
    return EvidenceCitation(
        source_url=fact.source_url,
        accession_number=fact.accession_number,
        locator=f"xbrl={fact.concept};period={fact.period_end.isoformat()}",
        excerpt=f"{fact.concept}={fact.value} {fact.unit}",
    )


def _insufficient(plan: QueryPlan, reason: str) -> FinancialAnswer:
    return FinancialAnswer(
        question=plan.query,
        status="insufficient_evidence",
        route=plan.intent,
        answer=f"Insufficient evidence: {reason}",
        plan=plan,
    )


def _format_decimal(value: Decimal) -> str:
    return f"{value.normalize():f}"


def _format_percentage(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.01')):f}"
