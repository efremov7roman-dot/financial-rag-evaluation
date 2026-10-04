"""Best-effort SEC filing HTML cleanup and section extraction."""

import re

from bs4 import BeautifulSoup

from financial_rag.errors import FilingParseError
from financial_rag.ingestion.models import ParsedDocument, ParsedSection

_WHITESPACE = re.compile(r"[ \t\xa0]+")
_ITEM_HEADING = re.compile(r"^item\s+\d{1,2}[a-z]?(?:\.|\s|$)", re.IGNORECASE)


def parse_filing(content: str) -> ParsedDocument:
    """Convert SEC HTML or plain text into clean text and practical item sections."""

    if not content.strip():
        raise FilingParseError("filing content is empty")

    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "noscript", "template"]):
        element.decompose()

    raw_lines = soup.get_text("\n").splitlines()
    lines = [_WHITESPACE.sub(" ", line).strip() for line in raw_lines]
    clean_lines = [line for line in lines if line]
    text = "\n".join(clean_lines).strip()
    if not text:
        raise FilingParseError("filing contains no usable text")

    sections = _extract_sections(clean_lines)
    if not sections:
        sections = (ParsedSection(name="Document", text=text),)
    return ParsedDocument(text=text, sections=sections)


def _extract_sections(lines: list[str]) -> tuple[ParsedSection, ...]:
    sections: list[ParsedSection] = []
    current_name = "Document"
    current_lines: list[str] = []

    for line in lines:
        is_heading = len(line) <= 160 and _ITEM_HEADING.match(line) is not None
        if is_heading:
            if current_lines:
                sections.append(ParsedSection(name=current_name, text="\n".join(current_lines)))
            current_name = line
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(ParsedSection(name=current_name, text="\n".join(current_lines)))
    return tuple(section for section in sections if section.text.strip())
