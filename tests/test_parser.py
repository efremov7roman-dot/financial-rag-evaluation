import pytest

from financial_rag.errors import FilingParseError
from financial_rag.ingestion.parser import parse_filing


def test_parser_removes_scripts_and_extracts_item_sections() -> None:
    document = parse_filing(
        """
        <html><body>
          <script>secret_noise()</script>
          <h2>Item 1. Business</h2><p>We sell useful products.</p>
          <h2>Item 1A. Risk Factors</h2><p>Demand may decline.</p>
        </body></html>
        """
    )

    assert "secret_noise" not in document.text
    assert [section.name for section in document.sections] == [
        "Item 1. Business",
        "Item 1A. Risk Factors",
    ]
    assert document.sections[1].text == "Demand may decline."


def test_parser_rejects_empty_content() -> None:
    with pytest.raises(FilingParseError, match="empty"):
        parse_filing("  ")
