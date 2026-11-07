"""Checks about missing data: real gaps, placeholder text, empty columns."""

from __future__ import annotations

import pandas as pd

from autoeda.checks.base import CheckContext, make_id, plural, severity_by_fraction
from autoeda.models import Fix, FixAction, Issue, IssueKind, Severity
from autoeda.types import DISGUISED_MISSING_TOKENS, ColumnKind, ColumnType

BLANK_LABEL = "<blank>"


def _missing_fix(name: str, info: ColumnType, fraction: float, drop_at: float) -> Fix | None:
    if fraction >= drop_at:
        return Fix(
            action=FixAction.DROP_COLUMN,
            column=name,
            description=f"Drop '{name}': most of its values are missing.",
        )
    if info.kind is ColumnKind.NUMERIC:
        return Fix(
            action=FixAction.IMPUTE_MEDIAN,
            column=name,
            description=f"Fill the missing values of '{name}' with its median.",
        )
    if info.kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
        return Fix(
            action=FixAction.IMPUTE_MODE,
            column=name,
            description=f"Fill the missing values of '{name}' with its most common value.",
        )
    return None


def check_empty_columns(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name in ctx.names_of_kind(ColumnKind.EMPTY):
        info = ctx.types[name]
        issues.append(
            Issue(
                id=make_id(IssueKind.EMPTY_COLUMN, name),
                kind=IssueKind.EMPTY_COLUMN,
                severity=Severity.HIGH,
                column=name,
                title="Empty column",
                message=f"Column '{name}' has no usable values ({info.reason}).",
                details={"fraction": 1.0},
                fix=Fix(
                    action=FixAction.DROP_COLUMN,
                    column=name,
                    safe_auto=True,
                    description=f"Drop the empty column '{name}'.",
                ),
            )
        )
    return issues


def check_missing(ctx: CheckContext) -> list[Issue]:
    issues = []
    rules = ctx.rules
    for name, info in ctx.types.items():
        if info.kind is ColumnKind.EMPTY or info.n_missing == 0:
            continue

        fraction = info.n_missing / info.n_rows
        issues.append(
            Issue(
                id=make_id(IssueKind.MISSING_VALUES, name),
                kind=IssueKind.MISSING_VALUES,
                severity=severity_by_fraction(fraction, rules.missing_medium, rules.missing_high),
                column=name,
                title="Missing values",
                message=(
                    f"Column '{name}' is missing {info.n_missing} of {info.n_rows} values "
                    f"({fraction:.1%})."
                ),
                details={"count": info.n_missing, "fraction": fraction},
                fix=_missing_fix(name, info, fraction, rules.missing_drop_column),
            )
        )
    return issues


def check_disguised_missing(ctx: CheckContext) -> list[Issue]:
    """Text such as "?", "unknown" or a blank string standing in for a missing value."""
    issues = []
    rules = ctx.rules
    for name, info in ctx.types.items():
        if info.kind is ColumnKind.EMPTY or not ctx.is_text_storage(name):
            continue

        values = ctx.df[name].dropna().astype(str).str.strip().str.lower()
        if values.empty:
            continue

        is_marker = values.isin(DISGUISED_MISSING_TOKENS) | (values == "")
        count = int(is_marker.sum())
        if count == 0:
            continue

        token_counts = values[is_marker].value_counts()
        tokens = sorted(str(token) for token in token_counts.index)
        shown = {
            (BLANK_LABEL if token == "" else str(token)): int(number)
            for token, number in token_counts.items()
        }

        fraction = count / ctx.n_rows
        severity = Severity.HIGH if fraction >= rules.disguised_high else Severity.MEDIUM
        shown_tokens = ", ".join(repr(token) for token in shown)
        issues.append(
            Issue(
                id=make_id(IssueKind.DISGUISED_MISSING, name),
                kind=IssueKind.DISGUISED_MISSING,
                severity=severity,
                column=name,
                title="Placeholder text used for missing values",
                message=(
                    f"Column '{name}' uses placeholder text for missing values in "
                    f"{plural(count, 'row')} ({fraction:.1%}): {shown_tokens}. If one of these "
                    "is a real category, keep it."
                ),
                details={"count": count, "fraction": fraction, "tokens": shown},
                fix=Fix(
                    action=FixAction.REPLACE_WITH_NAN,
                    column=name,
                    params={"tokens": tokens},
                    safe_auto=True,
                    description=f"Turn the placeholders in '{name}' into real missing values.",
                ),
            )
        )
    return issues


def check_missing_together(ctx: CheckContext) -> list[Issue]:
    """Columns whose gaps appear in the same rows (often one cause behind both)."""
    rules = ctx.rules
    names = [
        name
        for name, info in ctx.types.items()
        if info.kind is not ColumnKind.EMPTY
        and info.n_missing / info.n_rows >= rules.missing_together_min_fraction
    ][: rules.missing_together_max_columns]
    if len(names) < 2:
        return []

    mask = ctx.df[names].isna().astype(int)
    correlation = mask.corr()

    issues = []
    for first_position, first in enumerate(names):
        for second in names[first_position + 1 :]:
            value = correlation.loc[first, second]
            if pd.isna(value) or value < rules.missing_together_corr:
                continue
            issues.append(
                Issue(
                    id=make_id(IssueKind.MISSING_TOGETHER, first, second),
                    kind=IssueKind.MISSING_TOGETHER,
                    severity=Severity.INFO,
                    columns=[first, second],
                    title="Values go missing together",
                    message=(
                        f"'{first}' and '{second}' are missing in almost the same rows "
                        f"(agreement {float(value):.2f}); one cause may explain both."
                    ),
                    details={"correlation": float(value), "fraction": 0.0},
                )
            )
    return issues
