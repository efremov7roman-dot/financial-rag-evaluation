import json

import pytest

from financial_rag.logging import configure_logging, get_logger


def test_logging_emits_structured_json(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")

    get_logger(component="test").info("foundation_ready", phase=0)

    event = json.loads(capsys.readouterr().err)
    assert event["event"] == "foundation_ready"
    assert event["component"] == "test"
    assert event["phase"] == 0
    assert event["level"] == "info"
    assert event["timestamp"].endswith("Z")
