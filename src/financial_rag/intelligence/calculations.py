"""Deterministic Decimal-based financial calculations."""

from decimal import Decimal

from financial_rag.errors import FinancialIntelligenceError


def percent_change(current: Decimal, prior: Decimal) -> Decimal:
    """Calculate signed percentage change without binary floating-point drift."""

    if prior == 0:
        raise FinancialIntelligenceError("percent change is undefined when the prior value is zero")
    return (current - prior) / abs(prior) * Decimal(100)


def difference(current: Decimal, prior: Decimal) -> Decimal:
    """Calculate current minus prior value."""

    return current - prior


def ratio(numerator: Decimal, denominator: Decimal) -> Decimal:
    """Calculate a percentage ratio with an explicit zero guard."""

    if denominator == 0:
        raise FinancialIntelligenceError("ratio is undefined when the denominator is zero")
    return numerator / denominator * Decimal(100)
