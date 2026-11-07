"""Checks on numeric columns: sentinels, outliers, skew, correlated pairs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from autoeda.checks.base import CheckContext, make_id, plural
from autoeda.models import Fix, FixAction, Issue, IssueKind, Severity
from autoeda.rules import CheckRules
from autoeda.types import ColumnKind


def core_values(values: pd.Series, rules: CheckRules) -> pd.Series:
    """The values with sentinels and extreme outliers removed.

    Used for skew, so that one impossible value (an age of 250) is reported as an outlier
    and does not also make the whole column look skewed.
    """
    kept = values[~values.isin(find_sentinels(values, rules))]
    if len(kept) < rules.outlier_min_values:
        return kept

    q1, q3 = np.percentile(kept, [25, 75])
    spread = q3 - q1
    if spread == 0:
        return kept

    low = q1 - rules.iqr_extreme * spread
    high = q3 + rules.iqr_extreme * spread
    return kept[(kept >= low) & (kept <= high)]


def safe_skew(values: pd.Series) -> float:
    """Skewness, or 0.0 when there are too few distinct values to measure it."""
    if len(values) < 3 or values.nunique() < 3:
        return 0.0
    result = float(stats.skew(values))
    return 0.0 if np.isnan(result) else result


def find_sentinels(values: pd.Series, rules: CheckRules) -> list[float]:
    """Numbers like -999 or 9999 that are used as a code for "unknown".

    A number counts as a sentinel when it is on the list, appears often enough, and lies
    outside the range of the rest of the column. That last condition keeps a
    legitimate 999 in a price column from being flagged.
    """
    total = len(values)
    if total < rules.outlier_min_values:
        return []

    counts = values.value_counts()
    candidates = []
    for candidate in rules.sentinel_values:
        count = int(counts.get(candidate, 0))
        if count >= rules.sentinel_min_count and count / total >= rules.sentinel_min_fraction:
            candidates.append(candidate)
    if not candidates:
        return []

    rest = values[~values.isin(candidates)]
    if len(rest) < rules.outlier_min_values:
        return []

    q1, q3 = np.percentile(rest, [25, 75])
    spread = q3 - q1
    if spread == 0:
        return []

    lower = q1 - rules.iqr_mild * spread
    upper = q3 + rules.iqr_mild * spread
    lowest, highest = float(rest.min()), float(rest.max())
    return [
        value
        for value in candidates
        if value < lower or value > upper or value < lowest or value > highest
    ]


def check_sentinels(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name in ctx.names_of_kind(ColumnKind.NUMERIC):
        values = ctx.numeric_values(name)
        found = find_sentinels(values, ctx.rules)
        if not found:
            continue

        count = int(values.isin(found).sum())
        fraction = count / ctx.n_rows
        shown = ", ".join(f"{value:g}" for value in found)
        issues.append(
            Issue(
                id=make_id(IssueKind.SENTINEL_VALUES, name),
                kind=IssueKind.SENTINEL_VALUES,
                severity=Severity.MEDIUM,
                column=name,
                title="Possible placeholder numbers",
                message=(
                    f"Column '{name}' contains {shown} in {plural(count, 'row')} ({fraction:.1%}). "
                    "Numbers like this are often a code for 'unknown' and distort averages."
                ),
                details={"count": count, "fraction": fraction, "values": found},
                fix=Fix(
                    action=FixAction.REPLACE_WITH_NAN,
                    column=name,
                    params={"values": found},
                    description=f"Treat {shown} in '{name}' as missing values.",
                ),
            )
        )
    return issues


def check_outliers(ctx: CheckContext) -> list[Issue]:
    """Values far outside the usual range, using Tukey's fences on the IQR."""
    issues = []
    rules = ctx.rules
    for name in ctx.names_of_kind(ColumnKind.NUMERIC):
        values = ctx.numeric_values(name)
        sentinels = find_sentinels(values, rules)
        if sentinels:
            # Sentinels get their own issue; do not report them twice.
            values = values[~values.isin(sentinels)]

        total = len(values)
        if total < rules.outlier_min_values:
            continue

        q1, q3 = np.percentile(values, [25, 75])
        spread = q3 - q1
        if spread == 0:
            # More than half of the values are identical, so "far from the middle" has no
            # meaning here and every different value would be flagged.
            continue

        mild_low = q1 - rules.iqr_mild * spread
        mild_high = q3 + rules.iqr_mild * spread
        extreme_low = q1 - rules.iqr_extreme * spread
        extreme_high = q3 + rules.iqr_extreme * spread

        is_extreme = (values < extreme_low) | (values > extreme_high)
        is_outside = (values < mild_low) | (values > mild_high)
        extreme_count = int(is_extreme.sum())
        outside_count = int(is_outside.sum())

        if extreme_count == 0 and (
            outside_count / total < rules.outlier_mild_min_fraction
            or outside_count < rules.outlier_mild_min_count
        ):
            continue

        if extreme_count > 0:
            severity = (
                Severity.HIGH
                if extreme_count / total >= rules.outlier_high_fraction
                else Severity.MEDIUM
            )
            clip_low, clip_high = extreme_low, extreme_high
        else:
            severity = Severity.LOW
            clip_low, clip_high = mild_low, mild_high
        # A bound beyond the data changes nothing, so show the bound that actually bites.
        clip_low = max(clip_low, float(values.min()))
        clip_high = min(clip_high, float(values.max()))

        median = float(np.median(values))
        flagged = values[is_outside]
        ordered = flagged.reindex((flagged - median).abs().sort_values(ascending=False).index)
        examples = [float(value) for value in ordered.head(rules.example_limit)]
        # Skew of the typical values only, so the outliers themselves do not cause it.
        skew = safe_skew(values[~is_extreme])

        note = ""
        if abs(skew) >= 1.0:
            note = " The distribution is skewed, so some of these may be genuine."
        issues.append(
            Issue(
                id=make_id(IssueKind.OUTLIERS, name),
                kind=IssueKind.OUTLIERS,
                severity=severity,
                column=name,
                title="Outliers",
                message=(
                    f"Column '{name}' has {plural(outside_count, 'value')} outside the usual range "
                    f"[{mild_low:.4g}, {mild_high:.4g}], {extreme_count} of them extreme. "
                    f"Most unusual: {', '.join(f'{value:.6g}' for value in examples)}.{note}"
                ),
                details={
                    "count": outside_count,
                    "extreme_count": extreme_count,
                    "fraction": outside_count / total,
                    "lower_fence": float(mild_low),
                    "upper_fence": float(mild_high),
                    "extreme_lower_fence": float(extreme_low),
                    "extreme_upper_fence": float(extreme_high),
                    "examples": examples,
                    "skew": skew,
                    "flagged_index": [int(label) for label in flagged.index[:20]],
                },
                fix=Fix(
                    action=FixAction.CLIP_OUTLIERS,
                    column=name,
                    params={"lower": float(clip_low), "upper": float(clip_high)},
                    description=(
                        f"Clip '{name}' to [{clip_low:.4g}, {clip_high:.4g}]. "
                        "Check first that the values are errors and not real extremes."
                    ),
                ),
            )
        )
    return issues


def check_skew(ctx: CheckContext) -> list[Issue]:
    issues = []
    rules = ctx.rules
    for name in ctx.names_of_kind(ColumnKind.NUMERIC):
        values = core_values(ctx.numeric_values(name), rules)
        if len(values) < rules.skew_min_values:
            continue

        skew = safe_skew(values)
        if abs(skew) < rules.skew_info:
            continue

        severity = Severity.LOW if abs(skew) >= rules.skew_low else Severity.INFO
        fix = None
        if skew > 0 and float(values.min()) >= 0:
            fix = Fix(
                action=FixAction.LOG_TRANSFORM,
                column=name,
                description=f"Apply log(1 + x) to '{name}' to reduce the right skew.",
            )
        issues.append(
            Issue(
                id=make_id(IssueKind.SKEWED, name),
                kind=IssueKind.SKEWED,
                severity=severity,
                column=name,
                title="Skewed distribution",
                message=(
                    f"Column '{name}' is strongly skewed (skewness {skew:.2f}); the mean is "
                    "pulled toward the long tail and many models prefer a transformed column."
                ),
                details={"skew": skew, "fraction": 0.0},
                fix=fix,
            )
        )
    return issues


def check_correlation(ctx: CheckContext) -> list[Issue]:
    rules = ctx.rules
    names = ctx.names_of_kind(ColumnKind.NUMERIC)[: rules.max_corr_columns]
    if len(names) < 2:
        return []

    correlation = ctx.correlation(names)
    pairs = []
    for first_position, first in enumerate(names):
        for second in names[first_position + 1 :]:
            value = correlation.loc[first, second]
            if pd.isna(value) or abs(value) < rules.corr_low:
                continue
            pairs.append((first, second, float(value)))

    pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)

    issues = []
    for first, second, value in pairs[: rules.max_corr_pairs]:
        severity = Severity.MEDIUM if abs(value) >= rules.corr_high else Severity.LOW
        issues.append(
            Issue(
                id=make_id(IssueKind.HIGH_CORRELATION, first, second),
                kind=IssueKind.HIGH_CORRELATION,
                severity=severity,
                columns=[first, second],
                title="Nearly duplicate columns",
                message=(
                    f"'{first}' and '{second}' move almost together (correlation {value:.3f}). "
                    "They carry nearly the same information; consider keeping one."
                ),
                details={"correlation": value, "fraction": 0.0},
                fix=Fix(
                    action=FixAction.DROP_COLUMN,
                    column=second,
                    description=f"Drop '{second}' and keep '{first}'.",
                ),
            )
        )
    return issues
