"""Command line interface: `autoeda profile`, `autoeda clean`, `autoeda drift`."""

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from autoeda import __version__
from autoeda.models import Severity

app = typer.Typer(
    help="Find data quality problems in a CSV file and suggest fixes.",
    no_args_is_help=True,
    add_completion=False,
)


class FailOn(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


def _threshold(value: FailOn) -> Severity | None:
    return None if value is FailOn.none else Severity(value.value)


def _prepare_console() -> None:
    # Cell values can contain symbols a Windows console cannot show; print "?" instead of
    # crashing.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _version(show: bool) -> None:
    if show:
        typer.echo(f"autoeda {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    _prepare_console()


def _exists(path: Path) -> Path:
    if not path.is_file():
        typer.echo(f"Error: file not found: {path}", err=True)
        raise typer.Exit(2)
    return path


@app.command("profile")
def profile_command(
    file: Annotated[Path, typer.Argument(help="CSV file to examine.")],
    out: Annotated[Path | None, typer.Option(help="Write an HTML report here.")] = None,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Write the full report as JSON here.")
    ] = None,
    max_rows: Annotated[int | None, typer.Option(help="Only read the first N rows.")] = None,
    fail_on: Annotated[
        FailOn, typer.Option(help="Exit with code 1 if any issue is at least this severe.")
    ] = FailOn.none,
    offline: Annotated[
        bool, typer.Option(help="Embed the chart library so the HTML works without internet.")
    ] = False,
) -> None:
    """Profile a CSV file: list the problems found, most serious first."""
    from autoeda.profile import profile
    from autoeda.report import write_report

    report = profile(_exists(file), max_rows=max_rows)
    typer.echo(report.summary_text())

    if out:
        write_report(report, out, plotly_js="inline" if offline else "cdn")
        typer.echo(f"HTML report: {out}")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"JSON report: {json_out}")

    threshold = _threshold(fail_on)
    if threshold is not None and report.issues_at_least(threshold):
        raise typer.Exit(1)


@app.command("clean")
def clean_command(
    file: Annotated[Path, typer.Argument(help="CSV file to clean.")],
    out: Annotated[Path, typer.Option(help="Where to write the cleaned CSV.")],
    everything: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Also apply judgment calls: fill missing values, clip outliers, drop columns.",
        ),
    ] = False,
    skip: Annotated[list[str] | None, typer.Option(help="Issue id to leave alone.")] = None,
    script: Annotated[
        Path | None, typer.Option(help="Write the steps as a pandas script here.")
    ] = None,
    report: Annotated[Path | None, typer.Option(help="Write a before/after HTML report.")] = None,
    offline: Annotated[bool, typer.Option(help="Embed the chart library in the HTML.")] = False,
) -> None:
    """Clean a CSV file. By default only safe fixes are applied."""
    from autoeda.fixes import clean
    from autoeda.report import write_report

    result = clean(_exists(file), fixes="all" if everything else "safe", skip=skip)
    typer.echo(result.summary_text())

    out.parent.mkdir(parents=True, exist_ok=True)
    result.df.to_csv(out, index=False)
    typer.echo(f"Cleaned file: {out}")

    if script:
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(result.code, encoding="utf-8")
        typer.echo(f"Script: {script}")
    if report:
        write_report(
            result.before, report, cleaned=result, plotly_js="inline" if offline else "cdn"
        )
        typer.echo(f"HTML report: {report}")


@app.command("drift")
def drift_command(
    reference: Annotated[Path, typer.Argument(help="The earlier or training CSV.")],
    current: Annotated[Path, typer.Argument(help="The newer or live CSV.")],
    out: Annotated[Path | None, typer.Option(help="Write an HTML report here.")] = None,
    fail_on: Annotated[
        FailOn, typer.Option(help="Exit with code 1 if any column shifted at least this much.")
    ] = FailOn.none,
    offline: Annotated[bool, typer.Option(help="Embed the chart library in the HTML.")] = False,
) -> None:
    """Compare two CSV files and report which columns shifted."""
    from autoeda.drift import compare
    from autoeda.report import write_drift

    result = compare(_exists(reference), _exists(current))
    typer.echo(result.summary_text())

    if out:
        write_drift(result, out, plotly_js="inline" if offline else "cdn")
        typer.echo(f"HTML report: {out}")

    threshold = _threshold(fail_on)
    if threshold is not None and result.worst.rank >= threshold.rank:
        raise typer.Exit(1)


@app.command("ask")
def ask_command(
    file: Annotated[Path, typer.Argument(help="CSV file to ask about.")],
    question: Annotated[str, typer.Argument(help="Your question, in plain English.")],
    model: Annotated[str, typer.Option(help="Ollama model name.")] = "qwen2.5:7b-instruct",
    host: Annotated[str | None, typer.Option(help="Ollama server address.")] = None,
    no_clean: Annotated[
        bool, typer.Option("--no-clean", help="Query the raw table without cleaning it first.")
    ] = False,
    show_steps: Annotated[bool, typer.Option("--steps", help="Print what each step did.")] = False,
) -> None:
    """Ask a question about a CSV file. Needs a local Ollama server and the agent extras."""
    try:
        from autoeda.agent import OllamaClient, ask
        from autoeda.agent.llm import LLMError
    except ImportError:
        typer.echo('Install the agent extras first: pip install "autoeda[agent]"', err=True)
        raise typer.Exit(2) from None

    try:
        llm = OllamaClient(model=model, host=host)
    except LLMError as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(2) from None

    outcome = ask(_exists(file), question, llm, clean_data=not no_clean)
    if show_steps:
        for step in outcome.trace:
            typer.echo(f"  [{step['step']}] {step['detail']}")
        typer.echo("")
    typer.echo(outcome.answer)
    if outcome.result is None:
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
