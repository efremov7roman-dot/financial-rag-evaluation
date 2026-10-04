"""Structured logging configuration."""

import logging
from typing import Any, cast

import structlog
from structlog.stdlib import BoundLogger

from financial_rag.config import LogLevel


def configure_logging(level: LogLevel = "INFO") -> None:
    """Configure standard-library and structlog output as newline-delimited JSON."""

    logging.basicConfig(level=level, format="%(message)s", force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(sort_keys=True),
        ],
        wrapper_class=BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def get_logger(**context: Any) -> BoundLogger:
    """Return a structured logger bound to stable contextual fields."""

    return cast(BoundLogger, structlog.get_logger().bind(**context))
