"""Rate-limited SEC EDGAR metadata discovery and filing download."""

import time
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any, cast

import httpx
from bs4 import UnicodeDammit

from financial_rag.errors import SECDownloadError
from financial_rag.ingestion.models import DownloadedFiling, FilingRecord, FilingType
from financial_rag.ingestion.rate_limit import RateLimiter

SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def normalize_cik(cik: str) -> str:
    """Return the canonical ten-digit representation used by SEC submissions."""

    stripped = cik.strip()
    if not stripped.isdigit() or len(stripped) > 10:
        message = f"invalid CIK: {cik!r}"
        raise ValueError(message)
    return stripped.zfill(10)


class SECClient:
    """Small synchronous client that follows SEC identity and rate-limit guidance."""

    def __init__(
        self,
        *,
        user_agent: str,
        cache_dir: Path,
        requests_per_second: float = 5.0,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not user_agent.strip():
            message = "SEC user agent is required and must include application/contact information"
            raise ValueError(message)
        if max_retries < 0:
            message = "max_retries must not be negative"
            raise ValueError(message)
        self._cache_dir = cache_dir
        self._max_retries = max_retries
        self._sleep = sleep
        self._rate_limiter = RateLimiter(requests_per_second, sleep=sleep)
        self._owns_http_client = http_client is None
        self._headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }
        self._http_client = http_client or httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers=self._headers,
        )

    def __enter__(self) -> "SECClient":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the internally owned HTTP transport."""

        if self._owns_http_client:
            self._http_client.close()

    def discover_filings(
        self,
        cik: str,
        *,
        forms: Iterable[FilingType] = ("10-K", "10-Q"),
        limit: int | None = None,
    ) -> list[FilingRecord]:
        """Discover recent 10-K/10-Q filings from the SEC submissions endpoint."""

        normalized_cik = normalize_cik(cik)
        if limit is not None and limit < 1:
            message = "limit must be positive when provided"
            raise ValueError(message)
        allowed_forms = frozenset(forms)
        response = self._get(f"{SUBMISSIONS_BASE_URL}/CIK{normalized_cik}.json")
        try:
            payload = cast(dict[str, Any], response.json())
            company = str(payload["name"])
            tickers = cast(list[str], payload.get("tickers", []))
            ticker = tickers[0] if tickers else None
            filings = cast(dict[str, Any], payload["filings"])
            recent = cast(dict[str, list[str]], filings["recent"])
            rows = list(
                zip(
                    recent["form"],
                    recent["filingDate"],
                    recent["reportDate"],
                    recent["accessionNumber"],
                    recent["primaryDocument"],
                    strict=True,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            message = f"malformed SEC submissions response for CIK {normalized_cik}"
            raise SECDownloadError(message) from exc

        records: list[FilingRecord] = []
        for form, filing_date, report_date, accession, primary_document in rows:
            if form not in allowed_forms:
                continue
            accession_path = accession.replace("-", "")
            cik_path = str(int(normalized_cik))
            source_url = f"{ARCHIVES_BASE_URL}/{cik_path}/{accession_path}/{primary_document}"
            records.append(
                FilingRecord(
                    company=company,
                    ticker=ticker,
                    cik=normalized_cik,
                    filing_type=form,
                    filing_date=date.fromisoformat(filing_date),
                    report_date=date.fromisoformat(report_date) if report_date else None,
                    accession_number=accession,
                    primary_document=primary_document,
                    source_url=source_url,
                )
            )
            if limit is not None and len(records) >= limit:
                break
        return records

    def download_filing(self, filing: FilingRecord) -> DownloadedFiling:
        """Return filing content from immutable local cache or SEC EDGAR."""

        cache_path = (
            self._cache_dir
            / filing.cik
            / filing.accession_number.replace("-", "")
            / filing.primary_document
        )
        if cache_path.is_file():
            return DownloadedFiling(
                filing=filing,
                content=cache_path.read_text(encoding="utf-8"),
                cache_path=str(cache_path),
                from_cache=True,
            )

        response = self._get(filing.source_url)
        content = _decode_filing_content(response.content)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f"{cache_path.suffix}.tmp")
        temporary_path.write_text(content, encoding="utf-8")
        temporary_path.replace(cache_path)
        return DownloadedFiling(
            filing=filing,
            content=content,
            cache_path=str(cache_path),
            from_cache=False,
        )

    def _get(self, url: str) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            self._rate_limiter.wait()
            try:
                response = self._http_client.get(url, headers=self._headers)
            except httpx.RequestError as exc:
                last_error = exc
            else:
                if response.status_code not in RETRYABLE_STATUS_CODES:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPStatusError as exc:
                        message = f"SEC request returned {response.status_code}: {url}"
                        raise SECDownloadError(message) from exc
                    return response
                last_error = httpx.HTTPStatusError(
                    f"retryable SEC response {response.status_code}",
                    request=response.request,
                    response=response,
                )
            if attempt < self._max_retries:
                self._sleep(float(2**attempt))

        message = f"SEC request failed after {self._max_retries + 1} attempts: {url}"
        raise SECDownloadError(message) from last_error


def _decode_filing_content(content: bytes) -> str:
    """Detect SEC filing encoding before normalizing the immutable cache to UTF-8."""

    decoded = UnicodeDammit(content, is_html=True).unicode_markup
    if decoded is None:
        raise SECDownloadError("SEC filing content encoding could not be detected")
    return decoded
