from pathlib import Path

import httpx
import pytest

from financial_rag.api import create_app
from financial_rag.config import Settings
from financial_rag.errors import FinancialIntelligenceError
from financial_rag.intelligence.models import FinancialAnswer, MetadataFilter, QueryPlan
from financial_rag.runtime import ApplicationRuntime


class FakeRuntime:
    def __init__(self, *, ready: bool = True, fail: bool = False) -> None:
        self.loaded = False
        self._ready = ready
        self._fail = fail
        self.answer_calls = 0

    def readiness(self) -> dict[str, bool]:
        return {"artifacts": self._ready}

    def answer(self, question: str) -> FinancialAnswer:
        self.answer_calls += 1
        if self._fail:
            raise FinancialIntelligenceError("runtime failed")
        plan = QueryPlan(
            query=question,
            intent="structured_fact",
            filters=MetadataFilter(cik="0000000001"),
            years=(2024,),
            metric="revenue",
            concepts=("us-gaap:Revenue",),
        )
        return FinancialAnswer(
            question=question,
            status="insufficient_evidence",
            route="structured_fact",
            answer="Insufficient evidence: test fixture.",
            plan=plan,
        )


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_is_lazy_and_readiness_has_distinct_status() -> None:
    runtime = FakeRuntime(ready=False)
    transport = httpx.ASGITransport(app=create_app(runtime=runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        health = await client.get("/health")
        readiness = await client.get("/ready")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["ready"] is False
    assert readiness.status_code == 503
    assert readiness.json()["status"] == "not_ready"
    assert runtime.answer_calls == 0


@pytest.mark.anyio
async def test_query_returns_structured_trace_and_header() -> None:
    runtime = FakeRuntime()
    transport = httpx.ASGITransport(app=create_app(runtime=runtime))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/query",
            json={"question": "What was revenue in 2024?"},
            headers={"X-Request-ID": "portfolio-demo-1"},
        )

    assert response.status_code == 200
    assert response.headers["x-trace-id"] == "portfolio-demo-1"
    assert response.json()["trace"] == {
        "trace_id": "portfolio-demo-1",
        "started_at": response.json()["trace"]["started_at"],
        "duration_ms": response.json()["trace"]["duration_ms"],
        "route": "structured_fact",
        "status": "insufficient_evidence",
        "citation_count": 0,
    }
    assert runtime.answer_calls == 1


@pytest.mark.anyio
async def test_query_maps_runtime_failure_and_validates_payload() -> None:
    transport = httpx.ASGITransport(app=create_app(runtime=FakeRuntime(fail=True)))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        failed = await client.post("/v1/query", json={"question": "valid question"})
        invalid = await client.post("/v1/query", json={"question": "", "unexpected": True})

    assert failed.status_code == 503
    assert failed.json()["detail"]["code"] == "runtime_unavailable"
    assert invalid.status_code == 422


def test_runtime_readiness_checks_all_persisted_artifacts(tmp_path: Path) -> None:
    index_dir = tmp_path / "dense"
    index_dir.mkdir()
    for name in ("manifest.json", "documents.jsonl", "vectors.npy"):
        (index_dir / name).touch()
    facts = tmp_path / "facts.jsonl"
    facts.touch()
    runtime = ApplicationRuntime(Settings(index_dir=index_dir, fact_index_path=facts))

    assert all(runtime.readiness().values())
    assert runtime.loaded is False
