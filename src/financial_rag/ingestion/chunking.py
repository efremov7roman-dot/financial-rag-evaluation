"""Deterministic, section-aware filing chunking."""

import hashlib

from financial_rag.ingestion.models import (
    ChunkMetadata,
    DocumentChunk,
    FilingRecord,
    ParsedDocument,
)


def chunk_document(
    filing: FilingRecord,
    document: ParsedDocument,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[DocumentChunk]:
    """Split section text on whitespace and attach complete source provenance."""

    if max_chars < 1 or overlap_chars < 0 or overlap_chars >= max_chars:
        message = "chunk sizes require max_chars > overlap_chars >= 0"
        raise ValueError(message)

    chunks: list[DocumentChunk] = []
    for section in document.sections:
        for text in _split_text(section.text, max_chars=max_chars, overlap_chars=overlap_chars):
            index = len(chunks)
            identity = "\x1f".join(
                (filing.accession_number, section.name, str(index), text)
            ).encode()
            chunk_id = hashlib.sha256(identity).hexdigest()
            metadata = ChunkMetadata(
                company=filing.company,
                ticker=filing.ticker,
                cik=filing.cik,
                filing_type=filing.filing_type,
                filing_date=filing.filing_date,
                report_date=filing.report_date,
                section=section.name,
                source_url=filing.source_url,
                accession_number=filing.accession_number,
            )
            chunks.append(
                DocumentChunk(chunk_id=chunk_id, index=index, text=text, metadata=metadata)
            )
    return chunks


def _split_text(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        proposed_end = min(start + max_chars, len(normalized))
        end = proposed_end
        if proposed_end < len(normalized):
            boundary = normalized.rfind(" ", start + 1, proposed_end + 1)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(0, end - overlap_chars)
        if overlap_chars:
            following_space = normalized.find(" ", next_start)
            start = following_space + 1 if following_space != -1 and following_space < end else end
        else:
            start = end
    return chunks
