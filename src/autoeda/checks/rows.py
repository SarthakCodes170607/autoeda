"""Checks that look at whole rows: duplicates and unusual combinations of values."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm

from autoeda.checks.base import CheckContext, make_id, plural
from autoeda.models import Fix, FixAction, Issue, IssueKind, Severity
from autoeda.types import ColumnKind


def check_duplicates(ctx: CheckContext) -> list[Issue]:
    try:
        is_duplicate = ctx.df.duplicated(keep="first")
    except TypeError:
        # Some cell holds something unhashable (a list, say); rows cannot be compared.
        return []

    count = int(is_duplicate.sum())
    if count == 0:
        return []

    fraction = count / ctx.n_rows
    severity = Severity.HIGH if fraction >= ctx.rules.duplicate_high else Severity.MEDIUM
    examples = [int(label) for label in ctx.df.index[is_duplicate][: ctx.rules.example_limit]]
    return [
        Issue(
            id=make_id(IssueKind.DUPLICATE_ROWS, "rows"),
            kind=IssueKind.DUPLICATE_ROWS,
            severity=severity,
            title="Duplicate rows",
            message=(
                f"{plural(count, 'row')} ({fraction:.1%}) duplicate an earlier row exactly. "
                "Duplicates count the same event twice and inflate totals."
            ),
            details={"count": count, "fraction": fraction, "example_index": examples},
            fix=Fix(
                action=FixAction.DROP_DUPLICATE_ROWS,
                safe_auto=True,
                description="Drop exact duplicate rows, keeping the first copy.",
            ),
        )
    ]


def _without_near_duplicates(ctx: CheckContext, names: list[str]) -> list[str]:
    """Keep one column from each group of nearly identical columns.

    Two columns that move together (correlation 0.95 or more) are already reported as
    duplicates. Left in, they make ordinary tail values look unusual, because the tails of
    a tightly correlated pair are sparse regions.
    """
    if len(names) < 2:
        return names

    correlation = ctx.correlation(names).abs()
    kept: list[str] = []
    for name in names:
        is_copy = any(correlation.loc[name, other] >= ctx.rules.corr_high for other in kept)
        if not is_copy:
            kept.append(name)
    return kept


def _normal_scores(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map every column to a standard normal shape using the reference rows' ranks.

    The i-th smallest of n reference values maps to the normal quantile at (i + 0.5) / n.
    (scikit-learn's QuantileTransformer would send the single largest value in every
    column to about 5.2 standard deviations, which makes ordinary extremes look unusual.)
    """
    n = len(reference)
    levels = norm.ppf((np.arange(n) + 0.5) / n)
    result = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        result[:, column] = np.interp(values[:, column], np.sort(reference[:, column]), levels)
    return result


def _squared_distances(fit: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each scored row from the centre of the fit rows.

    Ranks bound every column, so a few odd rows cannot drag the covariance far and the
    plain estimate is enough. It uses a pseudo-inverse, so columns that are exact
    combinations of others do not break it.
    """
    from sklearn.covariance import EmpiricalCovariance

    return EmpiricalCovariance().fit(fit).mahalanobis(scored)


def check_multivariate_outliers(ctx: CheckContext) -> list[Issue]:
    """Rows whose combination of numbers is unusual even if each number alone is normal.

    Each column is first mapped to a normal shape using ranks, so skewed columns do not
    flood the result with ordinary tail values. scikit-learn's covariance estimators then
    give each row a Mahalanobis distance. Under a normal model that squared distance follows
    a chi-square distribution, so the cut-off is a chosen false alarm rate rather than a guess.
    """
    rules = ctx.rules
    names = ctx.names_of_kind(ColumnKind.NUMERIC)[: rules.multivariate_max_columns]
    names = _without_near_duplicates(ctx, names)
    if len(names) < 2 or ctx.n_rows < rules.multivariate_min_rows:
        return []

    frame = ctx.numeric_frame(names)
    frame = frame.dropna(axis=1, how="all")
    # Columns with only a few distinct values (ratings, counts) would turn into steps.
    frame = frame.loc[:, frame.nunique() >= rules.multivariate_min_unique]
    if frame.shape[1] < 2:
        return []
    frame = frame.fillna(frame.median())

    fit_frame = frame
    if len(frame) > rules.multivariate_fit_rows:
        fit_frame = frame.sample(rules.multivariate_fit_rows, random_state=0)

    scored_frame = frame
    sampled = False
    if len(frame) > rules.multivariate_max_rows_scored:
        scored_frame = frame.sample(rules.multivariate_max_rows_scored, random_state=0)
        sampled = True

    fit_normal = _normal_scores(fit_frame.to_numpy(), fit_frame.to_numpy())
    scored_normal = _normal_scores(fit_frame.to_numpy(), scored_frame.to_numpy())

    distance = _squared_distances(fit_normal, scored_normal)
    cutoff = float(chi2.ppf(1 - rules.multivariate_alpha, df=frame.shape[1]))
    flagged = distance > cutoff
    count = int(flagged.sum())
    if count == 0:
        return []

    fraction = count / len(scored_frame)
    if fraction > rules.multivariate_max_fraction:
        return []
    flagged_index = scored_frame.index[flagged]
    order = np.argsort(-distance[flagged])
    examples = [int(flagged_index[position]) for position in order[:10]]
    suffix = " Only a sample of the rows was scored." if sampled else ""
    return [
        Issue(
            id=make_id(IssueKind.MULTIVARIATE_OUTLIERS, "rows"),
            kind=IssueKind.MULTIVARIATE_OUTLIERS,
            severity=Severity.LOW,
            columns=list(frame.columns),
            title="Unusual rows",
            message=(
                f"{plural(count, 'row')} ({fraction:.1%}) combine numbers in an unusual way across "
                f"{frame.shape[1]} numeric columns, even if each value alone looks normal. "
                f"Worth a look, not necessarily errors.{suffix}"
            ),
            details={
                "count": count,
                "fraction": fraction,
                "example_index": examples,
                "flagged_index": [int(label) for label in flagged_index],
                "sampled": sampled,
                "cutoff": cutoff,
            },
        )
    ]
