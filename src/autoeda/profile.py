"""Profile a dataset: detect column types, run every check, summarise every column."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from autoeda.checks import ALL_CHECKS, CheckContext, run_checks
from autoeda.checks.base import Check, prepare_text
from autoeda.checks.categorical import guess_dayfirst
from autoeda.loader import load_csv
from autoeda.models import (
    ColumnProfile,
    CorrelationMatrix,
    Issue,
    Report,
    Severity,
)
from autoeda.rules import CheckRules
from autoeda.types import ColumnKind, TypeRules, infer_types

HISTOGRAM_MAX_BINS = 30
TOP_VALUES_SHOWN = 8
CORRELATION_MATRIX_MAX_COLUMNS = 30


def quality_score(issues: list[Issue], rules: CheckRules) -> float:
    """A rough 0-100 summary: 100 minus points for every issue, by severity.

    It is a relative indicator for comparing a dataset before and after cleaning, not a
    measure to compare between different datasets.
    """
    penalties = {
        Severity.HIGH: rules.penalty_high,
        Severity.MEDIUM: rules.penalty_medium,
        Severity.LOW: rules.penalty_low,
        Severity.INFO: 0.0,
    }
    total = sum(penalties[issue.severity] for issue in issues)
    return max(0.0, 100.0 - total)


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] == 0:
        raise ValueError("The table has no rows, so there is nothing to profile.")
    if df.shape[1] == 0:
        raise ValueError("The table has no columns, so there is nothing to profile.")

    # Column names become strings (issue ids are strings) and row labels become positions,
    # so every "row 17" in a report means the 18th row of the table.
    frame = df.rename(columns=str).reset_index(drop=True)
    duplicated = frame.columns[frame.columns.duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Column names must be unique; these appear more than once: {duplicated}")
    return frame


def _numeric_stats(values: pd.Series) -> tuple[dict[str, Any], dict[str, list[float]] | None]:
    count = len(values)
    if count == 0:
        return {}, None

    q1, median, q3 = (float(value) for value in np.percentile(values, [25, 50, 75]))
    summary: dict[str, Any] = {
        "count": count,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if count > 1 else None,
        "min": float(values.min()),
        "q1": q1,
        "median": median,
        "q3": q3,
        "max": float(values.max()),
    }
    if count >= 3 and values.nunique() >= 3:
        summary["skew"] = float(stats.skew(values))

    bins = int(min(HISTOGRAM_MAX_BINS, max(5, np.sqrt(count))))
    counts, edges = np.histogram(values, bins=bins)
    histogram = {
        "edges": [float(edge) for edge in edges],
        "counts": [float(number) for number in counts],
    }
    return summary, histogram


def _datetime_stats(ctx: CheckContext, name: str) -> dict[str, Any]:
    series = ctx.df[name]
    if ctx.types[name].stored_as_text:
        text = prepare_text(series)
        dayfirst, _ = guess_dayfirst(text, ctx.rules.default_dayfirst)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=dayfirst)
    else:
        parsed = series

    parsed = parsed.dropna()
    if parsed.empty:
        return {}

    earliest = parsed.min()
    latest = parsed.max()
    return {
        "min": earliest.isoformat(),
        "max": latest.isoformat(),
        "span_days": float((latest - earliest).days),
        "count": int(len(parsed)),
    }


def _column_profile(ctx: CheckContext, name: str) -> ColumnProfile:
    info = ctx.types[name]
    summary: dict[str, Any] = {}
    top_values: list[tuple[str, int]] = []
    histogram = None

    if info.kind is ColumnKind.NUMERIC:
        summary, histogram = _numeric_stats(ctx.numeric_values(name))
        if not info.stored_as_text:
            infinite = int(np.isinf(pd.to_numeric(ctx.df[name], errors="coerce")).sum())
            if infinite:
                summary["infinite"] = infinite
    elif info.kind is ColumnKind.DATETIME:
        summary = _datetime_stats(ctx, name)
    elif info.kind is not ColumnKind.EMPTY:
        values = ctx.df[name].dropna().astype(str)
        counts = values.value_counts().head(TOP_VALUES_SHOWN)
        top_values = [(str(label), int(number)) for label, number in counts.items()]
        if info.kind is ColumnKind.TEXT:
            summary["average_length"] = float(values.str.len().mean())

    return ColumnProfile(
        name=name,
        kind=info.kind,
        dtype=info.dtype,
        n_rows=info.n_rows,
        n_missing=info.n_missing,
        missing_fraction=info.n_missing / info.n_rows,
        n_unique=info.n_unique,
        stored_as_text=info.stored_as_text,
        reason=info.reason,
        stats=summary,
        top_values=top_values,
        histogram=histogram,
    )


def _correlation_matrix(ctx: CheckContext) -> CorrelationMatrix | None:
    names = ctx.names_of_kind(ColumnKind.NUMERIC)[:CORRELATION_MATRIX_MAX_COLUMNS]
    if len(names) < 2:
        return None

    matrix = ctx.correlation(names)
    rows = []
    for first in names:
        row = []
        for second in names:
            value = matrix.loc[first, second]
            row.append(None if pd.isna(value) else round(float(value), 4))
        rows.append(row)
    return CorrelationMatrix(labels=names, values=rows)


def profile(
    source: str | Path | pd.DataFrame,
    *,
    max_rows: int | None = None,
    type_rules: TypeRules | None = None,
    check_rules: CheckRules | None = None,
    checks: tuple[Check, ...] = ALL_CHECKS,
) -> Report:
    """Profile a CSV file or a DataFrame.

    Args:
        source: A path to a CSV file, or a DataFrame.
        max_rows: Look at no more than this many rows (the first ones).
        type_rules: Thresholds for column type detection.
        check_rules: Thresholds for the checks.
        checks: Which checks to run; all of them by default.
    """
    check_rules = check_rules if check_rules is not None else CheckRules()
    notes: list[str] = []
    source_name: str | None = None
    truncated = False

    if isinstance(source, pd.DataFrame):
        table = source
        if max_rows is not None and len(table) > max_rows:
            table = table.iloc[:max_rows]
            truncated = True
    else:
        loaded = load_csv(source, max_rows=max_rows)
        table = loaded.df
        truncated = loaded.truncated
        source_name = str(loaded.path)

    if truncated:
        notes.append(f"Only the first {len(table)} rows were analysed.")

    frame = prepare_frame(table)
    column_types = infer_types(frame, type_rules)
    ctx = CheckContext(
        df=frame,
        types={item.name: item for item in column_types},
        rules=check_rules,
    )

    issues = run_checks(ctx, checks)
    issues.sort(key=lambda item: (-item.severity.rank, -item.affected_fraction, item.id))

    return Report(
        source=source_name,
        n_rows=len(frame),
        n_cols=frame.shape[1],
        truncated=truncated,
        columns=[_column_profile(ctx, item.name) for item in column_types],
        issues=issues,
        quality_score=quality_score(issues, check_rules),
        correlation=_correlation_matrix(ctx),
        notes=notes,
    )
