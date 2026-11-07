"""Pick a chart for a query result. No model involved."""

from __future__ import annotations

import plotly.graph_objects as go

from autoeda.agent.queries import QueryResult
from autoeda.charts import ACCENT, _base


def result_figure(result: QueryResult) -> go.Figure | None:
    table = result.table
    spec = result.spec
    if len(table) < 2 or len(table.columns) < 2:
        return None

    if spec.operation == "trend":
        figure = go.Figure(
            go.Scatter(
                x=table.iloc[:, 0],
                y=table.iloc[:, 1],
                mode="lines+markers",
                line={"color": ACCENT},
            )
        )
        figure.update_yaxes(title_text=str(table.columns[1]))
        return _base(figure, 320, result.title)

    if spec.operation in ("aggregate", "top_k", "value_counts"):
        value_column = table.columns[-1]
        labels = table.iloc[:, : len(table.columns) - 1].astype(str).agg(" / ".join, axis=1)
        if table[value_column].dtype.kind not in "biuf":
            return None
        figure = go.Figure(
            go.Bar(
                x=table[value_column][::-1],
                y=labels[::-1],
                orientation="h",
                marker={"color": ACCENT},
            )
        )
        figure.update_xaxes(title_text=str(value_column))
        return _base(figure, max(220, 30 * len(table) + 80), result.title)
    return None
