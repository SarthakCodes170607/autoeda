import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from autoeda.cli import app

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_messy.csv"
runner = CliRunner()


def test_profile_prints_a_summary_and_exits_zero():
    result = runner.invoke(app, ["profile", str(SAMPLE)])

    assert result.exit_code == 0
    assert "Quality score" in result.output
    assert "empty_column:internal_flag" in result.output


def test_profile_writes_html_and_json(tmp_path):
    html = tmp_path / "out" / "report.html"
    data = tmp_path / "report.json"

    result = runner.invoke(
        app, ["profile", str(SAMPLE), "--out", str(html), "--json", str(data), "--offline"]
    )

    assert result.exit_code == 0
    assert "<title>Data quality report</title>" in html.read_text(encoding="utf-8")
    assert json.loads(data.read_text(encoding="utf-8"))["n_rows"] == 306


def test_fail_on_sets_the_exit_code():
    assert runner.invoke(app, ["profile", str(SAMPLE), "--fail-on", "high"]).exit_code == 1
    assert runner.invoke(app, ["profile", str(SAMPLE), "--fail-on", "none"]).exit_code == 0


def test_fail_on_passes_for_clean_data(tmp_path):
    path = tmp_path / "clean.csv"
    rng = np.random.default_rng(0)
    pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200)}).to_csv(path, index=False)

    assert runner.invoke(app, ["profile", str(path), "--fail-on", "low"]).exit_code == 0


def test_missing_file_gives_a_clean_error():
    result = runner.invoke(app, ["profile", "nope.csv"])

    assert result.exit_code == 2
    assert "file not found" in result.output


def test_max_rows_is_passed_on():
    result = runner.invoke(app, ["profile", str(SAMPLE), "--max-rows", "50"])

    assert "50 rows" in result.output


def test_clean_writes_csv_script_and_report(tmp_path):
    out, script, report = tmp_path / "c.csv", tmp_path / "s.py", tmp_path / "r.html"

    result = runner.invoke(
        app,
        ["clean", str(SAMPLE), "--out", str(out), "--script", str(script), "--report", str(report)],
    )

    assert result.exit_code == 0
    cleaned = pd.read_csv(out)
    assert len(cleaned) == 300
    assert "internal_flag" not in cleaned.columns
    assert "pd.read_csv" in script.read_text(encoding="utf-8")
    assert "After cleaning" in report.read_text(encoding="utf-8")


def test_clean_all_and_skip(tmp_path):
    out = tmp_path / "c.csv"

    result = runner.invoke(
        app, ["clean", str(SAMPLE), "--out", str(out), "--all", "--skip", "constant_column:country"]
    )

    assert result.exit_code == 0
    cleaned = pd.read_csv(out)
    assert "country" in cleaned.columns
    assert cleaned["unit_price"].isna().sum() == 0


def test_drift_command(tmp_path):
    rng = np.random.default_rng(0)
    ref, cur = tmp_path / "ref.csv", tmp_path / "cur.csv"
    pd.DataFrame({"x": rng.normal(0, 1, 500)}).to_csv(ref, index=False)
    pd.DataFrame({"x": rng.normal(3, 1, 500)}).to_csv(cur, index=False)
    html = tmp_path / "drift.html"

    result = runner.invoke(
        app, ["drift", str(ref), str(cur), "--out", str(html), "--fail-on", "high"]
    )

    assert result.exit_code == 1
    assert "[HIGH] x" in result.output
    assert "Data drift report" in html.read_text(encoding="utf-8")


def test_version_flag():
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("autoeda ")


def test_no_arguments_shows_help():
    result = runner.invoke(app, [])

    assert "profile" in result.output and "clean" in result.output


@pytest.mark.parametrize("command", ["profile", "clean", "drift"])
def test_every_command_has_help(command):
    assert runner.invoke(app, [command, "--help"]).exit_code == 0


def test_ask_reports_an_unreachable_model_without_a_traceback():
    # No Ollama server is running in the tests, so the call fails and says so.
    result = runner.invoke(
        app, ["ask", str(SAMPLE), "How many orders?", "--host", "http://127.0.0.1:1"]
    )

    assert result.exit_code == 1
    assert "could not answer" in result.output.lower()
    assert "Traceback" not in result.output
