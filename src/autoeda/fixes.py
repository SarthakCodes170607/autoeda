"""Apply the fixes the checks suggest, and show them as plain pandas code.

Two things live here and must always agree:

* `apply_fix` changes a DataFrame.
* `fix_to_code` writes the same change as a line of pandas that a person can read, paste
  and review. The tests run both on the same table and compare the results.

Nothing in the library executes generated code; `fix_to_code` only returns text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from autoeda.loader import load_csv
from autoeda.models import Fix, FixAction, Report
from autoeda.profile import prepare_frame, profile
from autoeda.rules import CheckRules
from autoeda.types import NUMBER_NOISE_PATTERN, TypeRules


class FixError(ValueError):
    """A fix cannot be applied to the table as it is now."""


# ---------------------------------------------------------------- order


def fix_phase(fix: Fix) -> int:
    """Position of a fix in the cleaning order (lower runs first).

    Placeholder text goes first so it is not parsed as data; spaces are stripped before
    numbers and dates are read; row and column drops come after values are repaired (two
    rows that differ only by a space become duplicates once stripped); anything that
    fills or reshapes values comes last because it depends on everything before it.
    """
    action = fix.action
    if action is FixAction.REPLACE_WITH_NAN:
        return 0 if "tokens" in fix.params else 5
    return {
        FixAction.STRIP_WHITESPACE: 1,
        FixAction.CONVERT_NUMERIC: 2,
        FixAction.CONVERT_DATETIME: 3,
        FixAction.NORMALIZE_CASE: 4,
        FixAction.DROP_DUPLICATE_ROWS: 6,
        FixAction.DROP_COLUMN: 7,
        FixAction.IMPUTE_MEDIAN: 8,
        FixAction.IMPUTE_MODE: 8,
        FixAction.CLIP_OUTLIERS: 9,
        FixAction.LOG_TRANSFORM: 10,
    }[action]


# ---------------------------------------------------------------- helpers


def _normalized_tokens(tokens: list[str]) -> list[str]:
    """Placeholder tokens as they are compared: trimmed and case-folded."""
    return sorted({token.strip().casefold() for token in tokens})


def _column(df: pd.DataFrame, fix: Fix) -> pd.Series:
    if fix.column is None:
        raise FixError(f"{fix.action.value} needs a column")
    if fix.column not in df.columns:
        raise FixError(f"column '{fix.column}' no longer exists")
    return df[fix.column]


def _is_text_only(series: pd.Series) -> bool:
    return bool(series.map(lambda value: isinstance(value, str) or pd.isna(value)).all())


def _require_numeric(series: pd.Series, fix: Fix) -> None:
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise FixError(
            f"column '{fix.column}' is not numeric yet, so {fix.action.value} is skipped"
        )


def _text(series: pd.Series) -> pd.Series:
    """The column as text, keeping missing values missing."""
    if _is_text_only(series):
        return series
    return series.map(lambda value: value if pd.isna(value) else str(value))


# ---------------------------------------------------------------- apply


def apply_fix(df: pd.DataFrame, fix: Fix) -> pd.DataFrame:
    """Return a copy of `df` with the fix applied. Raises FixError if it cannot be."""
    result = df.copy()
    action = fix.action

    if action is FixAction.DROP_DUPLICATE_ROWS:
        return result.drop_duplicates(keep="first").reset_index(drop=True)

    series = _column(result, fix)
    name = fix.column
    assert name is not None

    if action is FixAction.DROP_COLUMN:
        return result.drop(columns=[name])

    if action is FixAction.REPLACE_WITH_NAN:
        if "tokens" in fix.params:
            tokens = _normalized_tokens(fix.params["tokens"])
            matches = series.astype("string").str.strip().str.casefold().isin(tokens)
        else:
            values = [float(value) for value in fix.params["values"]]
            matches = pd.to_numeric(series, errors="coerce").isin(values)
        result[name] = series.mask(matches)

    elif action is FixAction.STRIP_WHITESPACE:
        if _is_text_only(series):
            result[name] = series.str.strip()
        else:
            result[name] = series.map(lambda v: v.strip() if isinstance(v, str) else v)

    elif action is FixAction.CONVERT_NUMERIC:
        cleaned = _text(series).str.replace(NUMBER_NOISE_PATTERN, "", regex=True)
        result[name] = pd.to_numeric(cleaned, errors="coerce")

    elif action is FixAction.CONVERT_DATETIME:
        dayfirst = bool(fix.params.get("dayfirst", False))
        result[name] = pd.to_datetime(
            _text(series).str.strip(), errors="coerce", format="mixed", dayfirst=dayfirst
        )

    elif action is FixAction.NORMALIZE_CASE:
        result[name] = series.replace(fix.params["mapping"])

    elif action is FixAction.IMPUTE_MEDIAN:
        _require_numeric(series, fix)
        result[name] = series.fillna(series.median())

    elif action is FixAction.IMPUTE_MODE:
        modes = series.mode()
        if modes.empty:
            raise FixError(f"column '{name}' has no values to take a mode from")
        result[name] = series.fillna(modes.iloc[0])

    elif action is FixAction.CLIP_OUTLIERS:
        _require_numeric(series, fix)
        result[name] = series.clip(lower=fix.params["lower"], upper=fix.params["upper"])

    elif action is FixAction.LOG_TRANSFORM:
        _require_numeric(series, fix)
        if bool((series <= -1).any()):
            raise FixError(f"column '{name}' has values of -1 or less, so log(1 + x) is undefined")
        result[name] = np.log1p(series)

    return result


# ---------------------------------------------------------------- code


def fix_to_code(fix: Fix) -> str:
    """The fix as pandas code that works on a DataFrame called `df`."""
    action = fix.action
    if action is FixAction.DROP_DUPLICATE_ROWS:
        return 'df = df.drop_duplicates(keep="first").reset_index(drop=True)'

    name = fix.column
    col = f"df[{name!r}]"

    if action is FixAction.DROP_COLUMN:
        return f"df = df.drop(columns=[{name!r}])"
    if action is FixAction.REPLACE_WITH_NAN:
        if "tokens" in fix.params:
            tokens = _normalized_tokens(fix.params["tokens"])
            lowered = f'{col}.astype("string").str.strip().str.casefold()'
            return f"{col} = {col}.mask({lowered}.isin({tokens!r}))"
        values = [float(value) for value in fix.params["values"]]
        return f'{col} = {col}.mask(pd.to_numeric({col}, errors="coerce").isin({values!r}))'
    if action is FixAction.STRIP_WHITESPACE:
        return f"{col} = {col}.str.strip()"
    if action is FixAction.CONVERT_NUMERIC:
        noise = f'{col}.str.replace(r"{NUMBER_NOISE_PATTERN}", "", regex=True)'
        return f'{col} = pd.to_numeric({noise}, errors="coerce")'
    if action is FixAction.CONVERT_DATETIME:
        dayfirst = bool(fix.params.get("dayfirst", False))
        return (
            f'{col} = pd.to_datetime({col}.str.strip(), errors="coerce", '
            f'format="mixed", dayfirst={dayfirst})'
        )
    if action is FixAction.NORMALIZE_CASE:
        return f"{col} = {col}.replace({fix.params['mapping']!r})"
    if action is FixAction.IMPUTE_MEDIAN:
        return f"{col} = {col}.fillna({col}.median())"
    if action is FixAction.IMPUTE_MODE:
        return f"{col} = {col}.fillna({col}.mode().iloc[0])"
    if action is FixAction.CLIP_OUTLIERS:
        lower, upper = fix.params["lower"], fix.params["upper"]
        return f"{col} = {col}.clip(lower={lower!r}, upper={upper!r})"
    if action is FixAction.LOG_TRANSFORM:
        return f"{col} = np.log1p({col})"
    raise AssertionError(f"unhandled action {action}")  # pragma: no cover


# ---------------------------------------------------------------- cleaning a table


@dataclass(frozen=True)
class AppliedFix:
    issue_id: str
    fix: Fix
    effect: str


@dataclass(frozen=True)
class SkippedFix:
    issue_id: str
    fix: Fix
    reason: str


@dataclass
class CleanResult:
    """What `clean` returns: the cleaned table, what was done, and a before/after check."""

    df: pd.DataFrame
    applied: list[AppliedFix]
    skipped: list[SkippedFix]
    before: Report
    after: Report
    source_name: str = "data.csv"
    _notes: list[str] = field(default_factory=list)

    @property
    def code(self) -> str:
        """A pandas script that repeats the applied fixes on the original file."""
        lines = ['"""Cleaning steps suggested by autoeda."""', ""]
        if any(item.fix.action is FixAction.LOG_TRANSFORM for item in self.applied):
            lines.append("import numpy as np")
        lines += ["import pandas as pd", "", f"df = pd.read_csv({self.source_name!r})", ""]
        for item in self.applied:
            lines.append(f"# {item.fix.description}")
            lines.append(fix_to_code(item.fix))
        lines.append("")
        return "\n".join(lines)

    def summary_text(self) -> str:
        lines = [
            f"Quality score: {self.before.quality_score:.0f} -> {self.after.quality_score:.0f}",
            f"Issues: {len(self.before.issues)} -> {len(self.after.issues)}",
            f"Applied {len(self.applied)} fixes, skipped {len(self.skipped)}.",
        ]
        lines += [f"  done: {item.fix.description} ({item.effect})" for item in self.applied]
        lines += [f"  skipped {item.issue_id}: {item.reason}" for item in self.skipped]
        return "\n".join(lines)


def _effect(before: pd.DataFrame, after: pd.DataFrame, fix: Fix) -> str:
    if fix.action is FixAction.DROP_DUPLICATE_ROWS:
        return f"removed {len(before) - len(after)} rows"
    if fix.action is FixAction.DROP_COLUMN:
        return "removed 1 column"
    name = fix.column
    assert name is not None
    old, new = before[name], after[name]
    changed = (old.astype("string").fillna("<NA>") != new.astype("string").fillna("<NA>")).sum()
    return f"changed {int(changed)} cells"


def clean(
    source: str | Path | pd.DataFrame,
    report: Report | None = None,
    *,
    fixes: Literal["safe", "all"] | list[str] = "safe",
    skip: list[str] | None = None,
    type_rules: TypeRules | None = None,
    check_rules: CheckRules | None = None,
) -> CleanResult:
    """Clean a table using the fixes found by `profile`.

    Args:
        source: A CSV path or DataFrame. The input is never modified.
        report: A report for the same table. Computed if not given.
        fixes: "safe" applies only fixes that need no judgment (stripping spaces, reading
            numbers and dates, merging spellings, dropping exact duplicates, dropping empty
            columns, turning placeholder text into missing values). "all" applies every
            suggestion, including filling, clipping and dropping columns. A list applies
            exactly the fixes of those issue ids.
        skip: Issue ids never to apply.
    """
    source_name = "data.csv"
    if isinstance(source, pd.DataFrame):
        frame = prepare_frame(source)
    else:
        source_name = Path(source).name
        frame = prepare_frame(load_csv(source).df)

    if report is None:
        report = profile(frame, type_rules=type_rules, check_rules=check_rules)

    if isinstance(fixes, list):
        fixable = {issue.id for issue in report.issues if issue.fix is not None}
        unknown = [issue_id for issue_id in fixes if issue_id not in fixable]
        if unknown:
            raise ValueError(f"No such issue id (or it has no fix): {unknown}")

    excluded = set(skip or [])
    chosen = []
    for issue in report.issues:
        if issue.fix is None or issue.id in excluded:
            continue
        if fixes == "safe" and not issue.fix.safe_auto:
            continue
        if isinstance(fixes, list) and issue.id not in fixes:
            continue
        chosen.append((issue.id, issue.fix))

    chosen.sort(key=lambda pair: fix_phase(pair[1]))  # stable: report order within a phase

    current = frame
    applied: list[AppliedFix] = []
    skipped: list[SkippedFix] = []
    for issue_id, fix in chosen:
        try:
            updated = apply_fix(current, fix)
        except FixError as error:
            skipped.append(SkippedFix(issue_id, fix, str(error)))
            continue
        applied.append(AppliedFix(issue_id, fix, _effect(current, updated, fix)))
        current = updated

    after = profile(current, type_rules=type_rules, check_rules=check_rules)
    return CleanResult(
        df=current,
        applied=applied,
        skipped=skipped,
        before=report,
        after=after,
        source_name=source_name,
    )
