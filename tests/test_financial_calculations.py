from decimal import Decimal

import pytest

from financial_rag.errors import FinancialIntelligenceError
from financial_rag.intelligence.calculations import difference, percent_change, ratio


def test_percent_change_uses_decimal_arithmetic() -> None:
    assert percent_change(Decimal("120"), Decimal("100")) == Decimal("20.0")
    assert percent_change(Decimal("80"), Decimal("100")) == Decimal("-20.0")


def test_difference_and_ratio_are_deterministic() -> None:
    assert difference(Decimal("12.3"), Decimal("10.1")) == Decimal("2.2")
    assert ratio(Decimal("45"), Decimal("100")) == Decimal("45.00")


@pytest.mark.parametrize("operation", [percent_change, ratio])
def test_calculations_reject_zero_denominator(operation: object) -> None:
    assert callable(operation)
    with pytest.raises(FinancialIntelligenceError, match="zero"):
        operation(Decimal("1"), Decimal("0"))
