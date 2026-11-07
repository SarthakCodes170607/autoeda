"""The checks. Each one looks at the data and returns a list of issues."""

from __future__ import annotations

from autoeda.checks.base import Check, CheckContext
from autoeda.checks.categorical import (
    check_case_variants,
    check_constant,
    check_date_as_text,
    check_high_cardinality,
    check_id_columns,
    check_imbalance,
    check_numeric_as_text,
    check_whitespace,
)
from autoeda.checks.missing import (
    check_disguised_missing,
    check_empty_columns,
    check_missing,
    check_missing_together,
)
from autoeda.checks.numeric import (
    check_correlation,
    check_outliers,
    check_sentinels,
    check_skew,
)
from autoeda.checks.rows import check_duplicates, check_multivariate_outliers
from autoeda.models import Issue

ALL_CHECKS: tuple[Check, ...] = (
    check_empty_columns,
    check_missing,
    check_disguised_missing,
    check_missing_together,
    check_constant,
    check_id_columns,
    check_numeric_as_text,
    check_date_as_text,
    check_whitespace,
    check_case_variants,
    check_sentinels,
    check_outliers,
    check_skew,
    check_correlation,
    check_imbalance,
    check_high_cardinality,
    check_duplicates,
    check_multivariate_outliers,
)


def run_checks(ctx: CheckContext, checks: tuple[Check, ...] = ALL_CHECKS) -> list[Issue]:
    issues: list[Issue] = []
    for check in checks:
        issues.extend(check(ctx))
    return issues


__all__ = [
    "ALL_CHECKS",
    "Check",
    "CheckContext",
    "check_duplicates",
    "run_checks",
]
