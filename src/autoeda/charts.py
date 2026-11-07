"""Plotly figures used by the HTML report and the Streamlit app.

All of them are drawn from numbers already stored in a Report (histogram bins, top
values, the correlation matrix), so no chart needs the original table.
"""

from __future__ import annotations

import plotly.graph_objects as go

from autoeda.drift import DriftReport
from autoeda.models import ColumnProfile, CorrelationMatrix, Report

ACCENT = "#3b6fb6"
MUTED = "#9aa5b1"
WARN = "#c2571a"


def _base(figure: go.Figure, height: int, title: str | None = None) -> go.Figure:
    figure.update_layout(
        template="simple_white",
        height=height,
        margin={"l": 50, "r": 20, "t": 40 if title else 15, "b": 40},
        title={"text": title, "x": 0, "font": {"size": 14}} if title else None,
        font={"family": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "size": 12},
        showlegend=False,
    )
    return figure


def histogram(column: ColumnProfile) -> go.Figure | None:
    """Distribution of a numeric column, drawn from its stored bins."""
    data = column.histogram
    if not data or not data["counts"]:
        return None
    edges, counts = data["edges"], data["counts"]
    centres = [(low + high) / 2 for low, high in zip(edges[:-1], edges[1:], strict=True)]
    widths = [high - low for low, high in zip(edges[:-1], edges[1:], strict=True)]
    figure = go.Figure(
        go.Bar(
            x=centres,
            y=counts,
            width=widths,
            marker={"color": ACCENT},
            hovertemplate="%{x:.4g}: %{y:.0f} rows<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="rows")
    return _base(figure, 220)


def top_values(column: ColumnProfile) -> go.Figure | None:
    """The most common values of a category-like column."""
    if not column.top_values:
        return None
    labels = [label if len(label) <= 28 else label[:27] + "…" for label, _ in column.top_values]
    counts = [count for _, count in column.top_values]
    figure = go.Figure(
        go.Bar(
            x=counts[::-1],
            y=labels[::-1],
            orientation="h",
            marker={"color": ACCENT},
            hovertemplate="%{y}: %{x} rows<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="rows")
    return _base(figure, max(160, 28 * len(labels) + 60))


def column_chart(column: ColumnProfile) -> go.Figure | None:
    """The right small chart for a column, or None if there is nothing to draw."""
    if column.histogram:
        return histogram(column)
    return top_values(column)


def missing_values(report: Report) -> go.Figure | None:
    """Share of missing values per column (only columns that have some)."""
    rows = [(item.name, item.missing_fraction) for item in report.columns if item.n_missing > 0]
    if not rows:
        return None
    rows.sort(key=lambda pair: pair[1])
    figure = go.Figure(
        go.Bar(
            x=[value * 100 for _, value in rows],
            y=[name for name, _ in rows],
            orientation="h",
            marker={"color": WARN},
            hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>",
        )
    )
    figure.update_xaxes(title_text="% of rows missing", range=[0, 100])
    return _base(figure, max(160, 26 * len(rows) + 70))


def correlation(matrix: CorrelationMatrix | None) -> go.Figure | None:
    if matrix is None or len(matrix.labels) < 2:
        return None
    figure = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=matrix.labels,
            y=matrix.labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            hovertemplate="%{x} vs %{y}: %{z:.2f}<extra></extra>",
            colorbar={"title": "r"},
        )
    )
    figure.update_yaxes(autorange="reversed")
    size = min(700, 120 + 38 * len(matrix.labels))
    return _base(figure, size)


def drift_scores(report: DriftReport) -> go.Figure | None:
    """PSI per column with the usual 0.10 and 0.25 thresholds marked."""
    rows = [(item.column, item.psi) for item in report.columns if item.psi is not None]
    if not rows:
        return None
    rows.sort(key=lambda pair: pair[1])
    colours = [WARN if value >= 0.25 else ACCENT if value >= 0.10 else MUTED for _, value in rows]
    figure = go.Figure(
        go.Bar(
            x=[value for _, value in rows],
            y=[name for name, _ in rows],
            orientation="h",
            marker={"color": colours},
            hovertemplate="%{y}: PSI %{x:.3f}<extra></extra>",
        )
    )
    for threshold in (0.10, 0.25):
        figure.add_vline(x=threshold, line_dash="dot", line_color=MUTED)
    figure.update_xaxes(title_text="population stability index (PSI)")
    return _base(figure, max(160, 26 * len(rows) + 70))
