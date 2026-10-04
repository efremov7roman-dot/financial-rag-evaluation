"""Transparent in-memory Okapi BM25 implementation."""

import math
import re
import time
from collections import Counter, defaultdict

import numpy as np

from financial_rag.errors import RetrievalError
from financial_rag.ingestion.models import DocumentChunk
from financial_rag.retrieval.models import SearchResponse, SearchResult

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokenize financial English text deterministically for lexical retrieval."""

    return _TOKEN.findall(text.casefold())


class BM25Index:
    """Exact Okapi BM25 index for a small development corpus."""

    def __init__(
        self, chunks: tuple[DocumentChunk, ...], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and 0 <= b <= 1")
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._document_lengths = np.asarray([len(tokenize(chunk.text)) for chunk in chunks])
        self._average_document_length = float(np.mean(self._document_lengths))
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            for term, frequency in Counter(tokenize(chunk.text)).items():
                postings[term].append((index, frequency))
        self._postings = dict(postings)

    def search(self, query: str, *, top_k: int) -> SearchResponse:
        """Return stable BM25 ranking with zero-score documents excluded."""

        if top_k < 1:
            raise RetrievalError("top_k must be positive")
        started_at = time.perf_counter()
        scores = np.zeros(len(self._chunks), dtype=np.float64)
        query_terms = set(tokenize(query))
        document_count = len(self._chunks)
        for term in query_terms:
            postings = self._postings.get(term, [])
            document_frequency = len(postings)
            if not document_frequency:
                continue
            inverse_document_frequency = math.log(
                1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in postings:
                length_ratio = (
                    self._document_lengths[document_index] / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                scores[document_index] += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1) / denominator
                )

        ranked_indices = [
            index for index in np.argsort(-scores, kind="stable").tolist() if scores[index] > 0
        ][:top_k]
        results = tuple(
            SearchResult(
                rank=rank,
                score=float(scores[index]),
                chunk=self._chunks[index],
            )
            for rank, index in enumerate(ranked_indices, start=1)
        )
        latency_ms = (time.perf_counter() - started_at) * 1_000
        return SearchResponse(
            query=query,
            latency_ms=latency_ms,
            stage_latencies_ms={"bm25": latency_ms},
            results=results,
        )
