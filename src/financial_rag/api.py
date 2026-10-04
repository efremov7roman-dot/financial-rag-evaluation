"""FastAPI surface for health checks and traced financial questions."""

import re
import time
from datetime import UTC, datetime
from typing import Protocol
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from financial_rag import __version__
from financial_rag.config import Settings, get_settings
from financial_rag.errors import FinancialRAGError
from financial_rag.intelligence.models import AnswerStatus, FinancialAnswer, QueryIntent
from financial_rag.logging import configure_logging, get_logger
from financial_rag.runtime import ApplicationRuntime

_TRACE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class Runtime(Protocol):
    """Minimal runtime contract used for API dependency injection."""

    @property
    def loaded(self) -> bool: ...

    def readiness(self) -> dict[str, bool]: ...

    def answer(self, question: str) -> FinancialAnswer: ...


class HealthResponse(BaseModel):
    """Liveness/readiness state without triggering model initialization."""

    model_config = ConfigDict(frozen=True)

    status: str
    version: str
    ready: bool
    runtime_loaded: bool
    checks: dict[str, bool]


class QueryRequest(BaseModel):
    """Validated public query payload."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2_000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("question must not be blank")
        return normalized


class QueryTrace(BaseModel):
    """Structured per-query operational trace returned and logged by the API."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    started_at: datetime
    duration_ms: float = Field(ge=0)
    route: QueryIntent
    status: AnswerStatus
    citation_count: int = Field(ge=0)


class QueryResponse(BaseModel):
    """Financial answer paired with its stable trace metadata."""

    model_config = ConfigDict(frozen=True)

    answer: FinancialAnswer
    trace: QueryTrace


def create_app(*, settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    """Create an injectable API application without eagerly loading ML models."""

    selected_settings = settings or get_settings()
    configure_logging(selected_settings.log_level)
    selected_runtime: Runtime = runtime or ApplicationRuntime(selected_settings)
    logger = get_logger(component="api")
    application = FastAPI(
        title="Financial RAG Retrieval & Evaluation System",
        version=__version__,
        description="Evidence-grounded retrieval and deterministic financial calculations.",
    )

    @application.get("/health", response_model=HealthResponse, tags=["operations"])
    def health() -> HealthResponse:
        checks = selected_runtime.readiness()
        return HealthResponse(
            status="ok",
            version=__version__,
            ready=all(checks.values()),
            runtime_loaded=selected_runtime.loaded,
            checks=checks,
        )

    @application.get("/ready", response_model=HealthResponse, tags=["operations"])
    def ready(response: Response) -> HealthResponse:
        checks = selected_runtime.readiness()
        is_ready = all(checks.values())
        if not is_ready:
            response.status_code = 503
        return HealthResponse(
            status="ready" if is_ready else "not_ready",
            version=__version__,
            ready=is_ready,
            runtime_loaded=selected_runtime.loaded,
            checks=checks,
        )

    @application.post("/v1/query", response_model=QueryResponse, tags=["query"])
    def query(
        payload: QueryRequest,
        response: Response,
        x_request_id: str | None = Header(default=None),
    ) -> QueryResponse:
        trace_id = _trace_id(x_request_id)
        response.headers["X-Trace-ID"] = trace_id
        started_at = datetime.now(UTC)
        started_counter = time.perf_counter()
        try:
            answer = selected_runtime.answer(payload.question)
        except (FinancialRAGError, OSError, ValueError) as exc:
            duration_ms = (time.perf_counter() - started_counter) * 1_000
            logger.error(
                "query_failed",
                trace_id=trace_id,
                duration_ms=duration_ms,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            raise HTTPException(
                status_code=503,
                detail={"code": "runtime_unavailable", "trace_id": trace_id},
            ) from exc
        trace = QueryTrace(
            trace_id=trace_id,
            started_at=started_at,
            duration_ms=(time.perf_counter() - started_counter) * 1_000,
            route=answer.route,
            status=answer.status,
            citation_count=len(answer.citations),
        )
        logger.info(
            "query_completed",
            **trace.model_dump(mode="json"),
            question_chars=len(payload.question),
        )
        return QueryResponse(answer=answer, trace=trace)

    return application


def _trace_id(value: str | None) -> str:
    if value is not None and _TRACE_ID.fullmatch(value):
        return value
    return uuid4().hex


app = create_app()
