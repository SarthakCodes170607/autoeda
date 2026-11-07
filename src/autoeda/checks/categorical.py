"""Checks on text-like and categorical columns, and on columns that should be another type."""

from __future__ import annotations

import re
import warnings

import pandas as pd

from autoeda.checks.base import CheckContext, make_id, plural, prepare_text
from autoeda.models import Fix, FixAction, Issue, IssueKind, Severity
from autoeda.types import (
    DISGUISED_MISSING_TOKENS,
    ColumnKind,
    parse_numeric_text,
)

DAY_MONTH_PATTERN = r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}"


def guess_dayfirst(values: pd.Series, default: bool = False) -> tuple[bool, bool]:
    """Decide whether dates like 03/04/2024 are day-first or month-first.

    Returns (dayfirst, ambiguous). A value such as 25/03/2024 can only be day-first, and
    03/25/2024 only month-first. When no value settles it, the answer is ambiguous and the
    default is used.
    """
    parts = values.astype(str).str.extract(DAY_MONTH_PATTERN).dropna()
    if parts.empty:
        return default, False

    first = pd.to_numeric(parts[0])
    second = pd.to_numeric(parts[1])
    day_first_proof = int((first > 12).sum())
    month_first_proof = int((second > 12).sum())

    if day_first_proof > 0 and month_first_proof == 0:
        return True, False
    if month_first_proof > 0 and day_first_proof == 0:
        return False, False
    return default, True


def check_constant(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name in ctx.names_of_kind(ColumnKind.CONSTANT):
        info = ctx.types[name]
        issues.append(
            Issue(
                id=make_id(IssueKind.CONSTANT_COLUMN, name),
                kind=IssueKind.CONSTANT_COLUMN,
                severity=Severity.LOW,
                column=name,
                title="Constant column",
                message=(
                    f"Column '{name}' never changes ({info.reason}). It carries no information "
                    "for analysis or modelling."
                ),
                details={"fraction": 1.0},
                fix=Fix(
                    action=FixAction.DROP_COLUMN,
                    column=name,
                    description=f"Drop the constant column '{name}'.",
                ),
            )
        )
    return issues


def check_id_columns(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name in ctx.names_of_kind(ColumnKind.ID):
        issues.append(
            Issue(
                id=make_id(IssueKind.ID_COLUMN, name),
                kind=IssueKind.ID_COLUMN,
                severity=Severity.INFO,
                column=name,
                title="Identifier column",
                message=(
                    f"Column '{name}' looks like an identifier. Leave it out of models and "
                    "of averages; use it only to find or join rows."
                ),
                details={"fraction": 0.0},
            )
        )
    return issues


def check_numeric_as_text(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name, info in ctx.types.items():
        if not (info.kind is ColumnKind.NUMERIC and info.stored_as_text):
            continue

        text = prepare_text(ctx.df[name])
        content = text[~text.str.lower().isin(DISGUISED_MISSING_TOKENS)]
        numbers = parse_numeric_text(content)
        unreadable = content[numbers.isna()]
        examples = list(dict.fromkeys(unreadable.head(50).tolist()))[: ctx.rules.example_limit]

        message = (
            f"Column '{name}' holds numbers stored as text (for example {text.iloc[0]!r}), "
            "so sums, averages and sorting do not work on it."
        )
        if len(unreadable) > 0:
            message += (
                f" {len(unreadable)} values are not numbers and would become missing: "
                f"{', '.join(repr(example) for example in examples)}."
            )
        issues.append(
            Issue(
                id=make_id(IssueKind.NUMERIC_AS_TEXT, name),
                kind=IssueKind.NUMERIC_AS_TEXT,
                severity=Severity.MEDIUM,
                column=name,
                title="Numbers stored as text",
                message=message,
                details={
                    "fraction": 1.0,
                    "parse_ratio": info.parse_ratio,
                    "unreadable_count": int(len(unreadable)),
                    "unreadable_examples": examples,
                },
                fix=Fix(
                    action=FixAction.CONVERT_NUMERIC,
                    column=name,
                    safe_auto=True,
                    description=(
                        f"Convert '{name}' to numbers (removing currency symbols, commas, "
                        "spaces and % signs)."
                    ),
                ),
            )
        )
    return issues


def check_date_as_text(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name, info in ctx.types.items():
        if not (info.kind is ColumnKind.DATETIME and info.stored_as_text):
            continue

        text = prepare_text(ctx.df[name])
        content = text[~text.str.lower().isin(DISGUISED_MISSING_TOKENS)]
        dayfirst, ambiguous = guess_dayfirst(content, ctx.rules.default_dayfirst)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(content, errors="coerce", format="mixed", dayfirst=dayfirst)
        unreadable = content[parsed.isna()]
        examples = list(dict.fromkeys(unreadable.head(50).tolist()))[: ctx.rules.example_limit]

        order = "day first" if dayfirst else "month first"
        message = f"Column '{name}' holds dates stored as text (for example {text.iloc[0]!r})."
        if ambiguous:
            message += (
                f" Dates such as 03/04/2024 are ambiguous here; {order} was assumed. "
                "Check this before relying on it."
            )
        elif re.search(DAY_MONTH_PATTERN, text.iloc[0]):
            message += f" The values are {order}."
        if len(unreadable) > 0:
            message += (
                f" {len(unreadable)} values are not dates: "
                f"{', '.join(repr(example) for example in examples)}."
            )

        issues.append(
            Issue(
                id=make_id(IssueKind.DATE_AS_TEXT, name),
                kind=IssueKind.DATE_AS_TEXT,
                severity=Severity.LOW,
                column=name,
                title="Dates stored as text",
                message=message,
                details={
                    "fraction": 1.0,
                    "dayfirst": dayfirst,
                    "ambiguous": ambiguous,
                    "unreadable_count": int(len(unreadable)),
                    "unreadable_examples": examples,
                },
                fix=Fix(
                    action=FixAction.CONVERT_DATETIME,
                    column=name,
                    params={"dayfirst": dayfirst},
                    safe_auto=True,
                    description=f"Convert '{name}' to real dates ({order}).",
                ),
            )
        )
    return issues


def check_whitespace(ctx: CheckContext) -> list[Issue]:
    issues = []
    for name, info in ctx.types.items():
        if info.kind is ColumnKind.EMPTY or not ctx.is_text_storage(name):
            continue

        values = ctx.df[name].dropna().astype(str)
        stripped = values.str.strip()
        padded = (values != stripped) & (stripped != "")
        count = int(padded.sum())
        if count == 0:
            continue

        examples = values[padded].head(ctx.rules.example_limit).tolist()
        fraction = count / ctx.n_rows
        issues.append(
            Issue(
                id=make_id(IssueKind.WHITESPACE, name),
                kind=IssueKind.WHITESPACE,
                severity=Severity.LOW,
                column=name,
                title="Extra spaces",
                message=(
                    f"Column '{name}' has leading or trailing spaces in "
                    f"{plural(count, 'value')} ({fraction:.1%}), for example {examples[0]!r}. "
                    "'Delhi' and 'Delhi ' count as different values."
                ),
                details={"count": count, "fraction": fraction, "examples": examples},
                fix=Fix(
                    action=FixAction.STRIP_WHITESPACE,
                    column=name,
                    safe_auto=True,
                    description=f"Remove leading and trailing spaces from '{name}'.",
                ),
            )
        )
    return issues


def check_case_variants(ctx: CheckContext) -> list[Issue]:
    """The same category written with different capital letters ("Delhi", "DELHI")."""
    issues = []
    rules = ctx.rules
    for name, info in ctx.types.items():
        if info.kind not in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
            continue
        if not ctx.is_text_storage(name) or info.n_unique > rules.case_variants_max_unique:
            continue

        values = ctx.df[name].dropna().astype(str)
        collapsed = values.str.replace(r"\s+", " ", regex=True).str.strip()
        collapsed = collapsed[collapsed != ""]
        if collapsed.empty:
            continue

        counts = collapsed.value_counts()
        by_key: dict[str, list[tuple[str, int]]] = {}
        for variant, count in counts.items():
            by_key.setdefault(str(variant).lower(), []).append((str(variant), int(count)))

        mapping: dict[str, str] = {}
        groups: list[list[str]] = []
        rows_changed = 0
        for variants in by_key.values():
            if len(variants) < 2:
                continue
            # Keep the most common spelling; ties are broken alphabetically for stability.
            ordered = sorted(variants, key=lambda item: (-item[1], item[0]))
            canonical = ordered[0][0]
            groups.append([variant for variant, _ in ordered])
            for variant, count in ordered[1:]:
                mapping[variant] = canonical
                rows_changed += count

        if not mapping:
            continue

        example = groups[0]
        fraction = rows_changed / ctx.n_rows
        issues.append(
            Issue(
                id=make_id(IssueKind.CASE_VARIANTS, name),
                kind=IssueKind.CASE_VARIANTS,
                severity=Severity.MEDIUM,
                column=name,
                title="Same category written differently",
                message=(
                    f"Column '{name}' spells the same category in different capitalisation "
                    f"({len(groups)} group(s), for example {', '.join(map(repr, example))}), "
                    "so grouping and counting split them apart."
                ),
                details={
                    "count": rows_changed,
                    "fraction": fraction,
                    "groups": groups[: rules.example_limit],
                },
                fix=Fix(
                    action=FixAction.NORMALIZE_CASE,
                    column=name,
                    params={"mapping": mapping},
                    safe_auto=True,
                    description=(
                        f"Rewrite each variant in '{name}' as its most common spelling "
                        "(also collapses repeated spaces)."
                    ),
                ),
            )
        )
    return issues


def check_imbalance(ctx: CheckContext) -> list[Issue]:
    issues = []
    rules = ctx.rules
    for name, info in ctx.types.items():
        if info.kind not in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN):
            continue
        if not 2 <= info.n_unique <= rules.imbalance_max_classes:
            continue

        shares = ctx.df[name].dropna().value_counts(normalize=True)
        minority_share = float(shares.min())
        if minority_share >= rules.imbalance_minority_low:
            continue

        minority = shares.idxmin()
        severity = (
            Severity.MEDIUM if minority_share < rules.imbalance_minority_medium else Severity.LOW
        )
        issues.append(
            Issue(
                id=make_id(IssueKind.CLASS_IMBALANCE, name),
                kind=IssueKind.CLASS_IMBALANCE,
                severity=severity,
                column=name,
                title="Rare category",
                message=(
                    f"In column '{name}', {minority!r} is only {minority_share:.1%} of the "
                    "rows. If you predict this column, accuracy will look good even for a "
                    "model that ignores that category."
                ),
                details={
                    "minority": str(minority),
                    "minority_share": minority_share,
                    "fraction": 0.0,
                },
            )
        )
    return issues


def check_high_cardinality(ctx: CheckContext) -> list[Issue]:
    issues = []
    limit = ctx.rules.high_cardinality_unique
    for name, info in ctx.types.items():
        if info.kind is not ColumnKind.CATEGORICAL or info.n_unique <= limit:
            continue
        issues.append(
            Issue(
                id=make_id(IssueKind.HIGH_CARDINALITY, name),
                kind=IssueKind.HIGH_CARDINALITY,
                severity=Severity.INFO,
                column=name,
                title="Many categories",
                message=(
                    f"Column '{name}' has {info.n_unique} different categories. Charts get "
                    "crowded and one-hot encoding becomes wide; consider grouping rare ones."
                ),
                details={"n_unique": info.n_unique, "fraction": 0.0},
            )
        )
    return issues
