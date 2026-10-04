import json

from typer.testing import CliRunner

from financial_rag import __version__
from financial_rag.cli import app

runner = CliRunner()


def test_info_command_reports_application_metadata() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "app": "financial-rag",
        "environment": "development",
        "version": __version__,
    }
