from pathlib import Path

import pytest

from financial_rag.config import Settings


def test_settings_have_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINANCIAL_RAG_ENVIRONMENT", raising=False)
    monkeypatch.delenv("FINANCIAL_RAG_LOG_LEVEL", raising=False)
    monkeypatch.delenv("FINANCIAL_RAG_DATA_DIR", raising=False)

    settings = Settings()

    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.data_dir == Path("data")


def test_settings_read_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINANCIAL_RAG_ENVIRONMENT", "test")
    monkeypatch.setenv("FINANCIAL_RAG_LOG_LEVEL", "DEBUG")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.log_level == "DEBUG"
