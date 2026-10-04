from pathlib import Path
from typing import Any

import httpx
import pytest

from financial_rag.errors import SECDownloadError
from financial_rag.ingestion.sec_client import SECClient, _decode_filing_content, normalize_cik


def _submissions_payload() -> dict[str, Any]:
    return {
        "name": "Example Corp",
        "tickers": ["EXM"],
        "filings": {
            "recent": {
                "form": ["8-K", "10-K", "10-Q"],
                "filingDate": ["2025-03-01", "2025-02-15", "2024-11-01"],
                "reportDate": ["2025-03-01", "2024-12-31", "2024-09-30"],
                "accessionNumber": [
                    "0000320193-25-000003",
                    "0000320193-25-000002",
                    "0000320193-24-000001",
                ],
                "primaryDocument": ["eight-k.htm", "annual.htm", "quarterly.htm"],
            }
        },
    }


def test_discover_and_cache_filing(tmp_path: Path) -> None:
    filing_requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal filing_requests
        assert request.headers["user-agent"] == "FinancialRAG test@example.com"
        if request.url.host == "data.sec.gov":
            return httpx.Response(200, request=request, json=_submissions_payload())
        filing_requests += 1
        return httpx.Response(
            200,
            request=request,
            text="<html><body>Item 1. Business</body></html>",
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = SECClient(
        user_agent="FinancialRAG test@example.com",
        cache_dir=tmp_path,
        requests_per_second=100_000,
        http_client=http_client,
    )

    filings = client.discover_filings("320193", forms=("10-K",), limit=1)
    first = client.download_filing(filings[0])
    second = client.download_filing(filings[0])

    assert len(filings) == 1
    assert filings[0].cik == "0000320193"
    assert filings[0].ticker == "EXM"
    assert filings[0].source_url.endswith("/320193/000032019325000002/annual.htm")
    assert first.from_cache is False
    assert second.from_cache is True
    assert second.content == first.content
    assert filing_requests == 1
    assert Path(first.cache_path).is_file()
    http_client.close()


def test_normalize_cik_rejects_non_numeric_input() -> None:
    assert normalize_cik(" 320193 ") == "0000320193"

    try:
        normalize_cik("not-a-cik")
    except ValueError as exc:
        assert "invalid CIK" in str(exc)
    else:
        raise AssertionError("invalid CIK must be rejected")


def test_retryable_response_is_retried_with_backoff(tmp_path: Path) -> None:
    request_count = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(429, request=request)
        return httpx.Response(200, request=request, json=_submissions_payload())

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = SECClient(
        user_agent="FinancialRAG test@example.com",
        cache_dir=tmp_path,
        requests_per_second=100_000,
        max_retries=1,
        http_client=http_client,
        sleep=sleeps.append,
    )

    assert len(client.discover_filings("320193", forms=("10-K",), limit=1)) == 1
    assert request_count == 2
    assert 1.0 in sleeps
    http_client.close()


def test_non_retryable_response_fails_immediately(tmp_path: Path) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(404, request=request)

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = SECClient(
        user_agent="FinancialRAG test@example.com",
        cache_dir=tmp_path,
        max_retries=3,
        http_client=http_client,
    )

    with pytest.raises(SECDownloadError, match="404"):
        client.discover_filings("320193")
    assert request_count == 1
    http_client.close()


def test_filing_decoder_preserves_windows_1252_punctuation() -> None:
    content = _decode_filing_content(b"<html><body>Company\x92s non\x96U.S. sales</body></html>")

    assert "Company\u2019s non\u2013U.S. sales" in content
    assert "�" not in content
