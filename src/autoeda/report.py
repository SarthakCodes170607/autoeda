"""Render a Report (and optionally a cleaning result or a drift comparison) as HTML."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import jinja2
from plotly.graph_objects import Figure

from autoeda import charts
from autoeda.drift import DriftReport
from autoeda.fixes import CleanResult, fix_to_code
from autoeda.models import Report, Severity

PlotlyJS = Literal["cdn", "inline"]

_environment = jinja2.Environment(
    loader=jinja2.PackageLoader("autoeda", "templates"),
    autoescape=jinja2.select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


class _Figures:
    """Turns figures into HTML fragments. The Plotly script is added once, by the page."""

    def __init__(self) -> None:
        self.count = 0

    def html(self, figure: Figure | None) -> str:
        if figure is None:
            return ""
        self.count += 1
        return figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False, "responsive": True},
        )


def _plotly_script(mode: PlotlyJS) -> str:
    if mode == "inline":
        from plotly.offline import get_plotlyjs

        return f"<script>{get_plotlyjs()}</script>"
    from plotly.offline import get_plotlyjs_version

    url = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    return f'<script src="{url}" charset="utf-8"></script>'


def _score_label(score: float) -> str:
    if score >= 90:
        return "good"
    if score >= 70:
        return "needs some work"
    if score >= 50:
        return "poor"
    return "very poor"


def render_report(
    report: Report,
    *,
    cleaned: CleanResult | None = None,
    plotly_js: PlotlyJS = "cdn",
    title: str = "Data quality report",
) -> str:
    """The report as one HTML page.

    Args:
        plotly_js: "cdn" keeps the file small and needs internet to draw charts;
            "inline" embeds the library (about 4 MB) so the file works offline.
    """
    figures = _Figures()
    column_charts = {item.name: figures.html(charts.column_chart(item)) for item in report.columns}
    issue_code = {item.id: fix_to_code(item.fix) for item in report.issues if item.fix is not None}
    template = _environment.get_template("report.html.j2")
    return template.render(
        title=title,
        report=report,
        cleaned=cleaned,
        severities=[Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO],
        counts=report.severity_counts(),
        score_label=_score_label(report.quality_score),
        after_label=_score_label(cleaned.after.quality_score) if cleaned else "",
        column_charts=column_charts,
        missing_chart=figures.html(charts.missing_values(report)),
        correlation_chart=figures.html(charts.correlation(report.correlation)),
        issue_code=issue_code,
        plotly_script=_plotly_script(plotly_js),
    )


def render_drift(
    drift: DriftReport, *, plotly_js: PlotlyJS = "cdn", title: str = "Data drift report"
) -> str:
    figures = _Figures()
    template = _environment.get_template("drift.html.j2")
    return template.render(
        title=title,
        drift=drift,
        chart=figures.html(charts.drift_scores(drift)),
        plotly_script=_plotly_script(plotly_js),
    )


def write_report(report: Report, path: str | Path, **options) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(report, **options), encoding="utf-8")
    return target


def write_drift(drift: DriftReport, path: str | Path, **options) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_drift(drift, **options), encoding="utf-8")
    return target
