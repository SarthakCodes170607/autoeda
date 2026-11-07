"""A small, validated query language that the language model fills in.

The model never writes code. It chooses an operation and column names; this module checks
them against the real table and runs ordinary pandas operations.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

MAX_ROWS = 50

Operation = Literal["aggregate", "top_k", "describe", "correlation", "value_counts", "trend"]
Aggregation = Literal["sum", "mean", "median", "min", "max", "count", "nunique"]
FilterOp = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains", "not_null"]


class QueryError(ValueError):
    """The query does not make sense for this table. The message is shown to the model."""


class Filter(BaseModel):
    column: str
    op: FilterOp
    value: Any = None


class QuerySpec(BaseModel):
    """One question about a table, as structured data."""

    operation: Operation = Field(
        description=(
            "aggregate: one number, or one number per group_by group. "
            "top_k: the k largest (or smallest) groups by an aggregated column. "
            "describe: summary statistics of numeric columns. "
            "correlation: correlation between numeric columns. "
            "value_counts: how often each value of a column occurs. "
            "trend: an aggregated column over time."
        )
    )
    column: str | None = Field(
        default=None, description="Column to aggregate, rank or count values of."
    )
    agg: Aggregation = "sum"
    group_by: list[str] = Field(default_factory=list)
    k: int = Field(default=5, ge=1, le=MAX_ROWS)
    ascending: bool = False
    columns: list[str] = Field(default_factory=list, description="For describe and correlation.")
    date_column: str | None = Field(default=None, description="For trend.")
    freq: Literal["D", "W", "M", "Q", "Y"] = "M"
    filters: list[Filter] = Field(default_factory=list)


class QueryResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    spec: QuerySpec
    title: str
    table: pd.DataFrame
    rows_used: int

    def to_markdown(self, max_rows: int = 15) -> str:
        shown = self.table.head(max_rows)
        header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
        divider = "|" + "|".join("---" for _ in shown.columns) + "|"
        body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in shown.to_numpy()]
        return "\n".join([header, divider, *body])


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


# ---------------------------------------------------------------- validation helpers


def _require_column(df: pd.DataFrame, name: str | None, role: str) -> str:
    if not name:
        raise QueryError(f"{role} is required")
    if name not in df.columns:
        close = [c for c in df.columns if c.lower() == name.lower()]
        hint = f" Did you mean '{close[0]}'?" if close else ""
        raise QueryError(f"There is no column '{name}'.{hint} Columns: {list(df.columns)}")
    return name


def _require_numeric(df: pd.DataFrame, name: str) -> None:
    if not pd.api.types.is_numeric_dtype(df[name]) or pd.api.types.is_bool_dtype(df[name]):
        raise QueryError(f"Column '{name}' is not numeric, so it cannot be summed or averaged")


def _apply_filter(df: pd.DataFrame, item: Filter) -> pd.Series:
    name = _require_column(df, item.column, "filter column")
    series = df[name]
    op, value = item.op, item.value

    if op == "not_null":
        return series.notna()
    if value is None:
        raise QueryError(f"Filter on '{name}' with '{op}' needs a value")

    is_date = pd.api.types.is_datetime64_any_dtype(series)
    is_number = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)

    def coerce(raw: Any) -> Any:
        try:
            if is_date:
                return pd.Timestamp(raw)
            if is_number:
                return float(raw)
        except (TypeError, ValueError) as error:
            raise QueryError(f"Cannot compare column '{name}' with {raw!r}") from error
        return str(raw)

    if op in ("in", "not_in"):
        values = value if isinstance(value, list) else [value]
        mask = series.isin([coerce(v) for v in values])
        return mask if op == "in" else ~mask
    if op == "contains":
        return (
            series.astype("string").str.contains(str(value), case=False, regex=False).fillna(False)
        )

    target = coerce(value)
    comparisons = {
        "==": series == target,
        "!=": series != target,
        ">": series > target,
        ">=": series >= target,
        "<": series < target,
        "<=": series <= target,
    }
    # Missing values never satisfy a comparison, including "!=" (as in SQL).
    return comparisons[op].fillna(False) & series.notna()


# ---------------------------------------------------------------- execution


def run_query(df: pd.DataFrame, spec: QuerySpec) -> QueryResult:
    """Run a validated query. Raises QueryError with a helpful message if it is invalid."""
    frame = df
    for item in spec.filters:
        frame = frame[_apply_filter(frame, item)]
    if frame.empty:
        raise QueryError("No rows match the filters")
    title, table = _dispatch(frame, spec)
    return QueryResult(spec=spec, title=title, table=table.head(MAX_ROWS), rows_used=len(frame))


def _dispatch(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    operation = spec.operation
    if operation in ("aggregate", "top_k"):
        return _aggregate(frame, spec)
    if operation == "describe":
        return _describe(frame, spec)
    if operation == "correlation":
        return _correlation(frame, spec)
    if operation == "value_counts":
        return _value_counts(frame, spec)
    return _trend(frame, spec)


def _aggregate(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    metric = _require_column(frame, spec.column, "column")
    groups = [_require_column(frame, name, "group_by column") for name in spec.group_by]
    if spec.agg not in ("count", "nunique"):
        _require_numeric(frame, metric)
    label = f"{spec.agg} of {metric}"

    if groups:
        table = frame.groupby(groups, dropna=True)[metric].agg(spec.agg).reset_index()
        table = table.rename(columns={metric: label})
        table = table.sort_values(label, ascending=spec.ascending, kind="stable")
        if spec.operation == "top_k":
            table = table.head(spec.k)
        title = f"{label} by {', '.join(groups)}"
    else:
        if spec.operation == "top_k":
            if spec.agg in ("count", "nunique"):
                raise QueryError("top_k without group_by ranks rows by a numeric column")
            table = (
                frame.nlargest(spec.k, metric)
                if not spec.ascending
                else frame.nsmallest(spec.k, metric)
            )
            table = table.reset_index(drop=True)
            return f"rows with the {'smallest' if spec.ascending else 'largest'} {metric}", table
        value = frame[metric].agg(spec.agg)
        table = pd.DataFrame({label: [value]})
        title = label
    return title, table.reset_index(drop=True)


def _describe(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    names = [_require_column(frame, name, "columns") for name in spec.columns]
    if not names:
        names = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    for name in names:
        _require_numeric(frame, name)
    if not names:
        raise QueryError("There are no numeric columns to describe")
    table = frame[names].describe().T.reset_index().rename(columns={"index": "column"})
    return "summary statistics", table


def _correlation(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    names = [_require_column(frame, name, "columns") for name in spec.columns]
    if len(names) < 2:
        raise QueryError("correlation needs at least two columns")
    for name in names:
        _require_numeric(frame, name)
    table = frame[names].corr().round(4).reset_index().rename(columns={"index": "column"})
    return "correlation", table


def _value_counts(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    name = _require_column(frame, spec.column, "column")
    counts = frame[name].value_counts(dropna=True).head(spec.k if spec.k else MAX_ROWS)
    table = counts.rename_axis(name).reset_index(name="count")
    return f"how often each {name} occurs", table


def _trend(frame: pd.DataFrame, spec: QuerySpec) -> tuple[str, pd.DataFrame]:
    date = _require_column(frame, spec.date_column, "date_column")
    metric = _require_column(frame, spec.column, "column")
    if not pd.api.types.is_datetime64_any_dtype(frame[date]):
        raise QueryError(f"Column '{date}' is not a date column")
    if spec.agg not in ("count", "nunique"):
        _require_numeric(frame, metric)
    label = f"{spec.agg} of {metric}"
    period = frame[date].dt.to_period(spec.freq)
    table = frame.groupby(period)[metric].agg(spec.agg).reset_index()
    table.columns = ["period", label]
    table["period"] = table["period"].astype(str)
    return f"{label} per {spec.freq}", table
