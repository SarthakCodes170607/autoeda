"""Compare two versions of a table and report what shifted.

Typical use: yesterday's data against today's, or training data against live data.
The main number is the Population Stability Index (PSI), the usual choice for this in
credit scoring and model monitoring: below 0.10 is stable, 0.10 to 0.25 is a moderate
shift, above 0.25 is a large one. The Kolmogorov-Smirnov statistic is reported too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats

from autoeda.checks.base import CheckContext
from autoeda.loader import load_csv
from autoeda.models import Severity
from autoeda.profile import prepare_frame
from autoeda.rules import CheckRules
from autoeda.types import ColumnKind, TypeRules, infer_types

PSI_MODERATE = 0.10
PSI_LARGE = 0.25
MISSING_SHIFT_MODERATE = 0.10
MISSING_SHIFT_LARGE = 0.25
MIN_VALUES = 20
MAX_CATEGORIES = 50
BINS = 10
EPSILON = 1e-4


class ColumnDrift(BaseModel):
    column: str
    kind: ColumnKind
    severity: Severity
    message: str
    psi: float | None = None
    ks_statistic: float | None = None
    ks_pvalue: float | None = None
    reference_mean: float | None = None
    current_mean: float | None = None
    mean_shift_in_std: float | None = None
    reference_missing: float
    current_missing: float
    new_categories: list[str] = Field(default_factory=list)
    gone_categories: list[str] = Field(default_factory=list)


class DriftReport(BaseModel):
    reference_rows: int
    current_rows: int
    columns: list[ColumnDrift]
    added_columns: list[str] = Field(default_factory=list)
    removed_columns: list[str] = Field(default_factory=list)
    type_changes: list[tuple[str, str, str]] = Field(default_factory=list)

    @property
    def worst(self) -> Severity:
        levels = [item.severity for item in self.columns]
        if self.added_columns or self.removed_columns or self.type_changes:
            levels.append(Severity.HIGH)
        return max(levels, key=lambda level: level.rank, default=Severity.INFO)

    def drifted(self, minimum: Severity = Severity.MEDIUM) -> list[ColumnDrift]:
        return [item for item in self.columns if item.severity.rank >= minimum.rank]

    def summary_text(self) -> str:
        lines = [
            f"Reference: {self.reference_rows} rows. Current: {self.current_rows} rows.",
        ]
        if self.removed_columns:
            lines.append(
                f"Columns missing from the current data: {', '.join(self.removed_columns)}"
            )
        if self.added_columns:
            lines.append(f"New columns in the current data: {', '.join(self.added_columns)}")
        for name, before, after in self.type_changes:
            lines.append(f"Type of '{name}' changed from {before} to {after}")
        drifted = self.drifted()
        if not drifted:
            lines.append("No column shifted by more than a moderate amount.")
        for item in drifted:
            lines.append(f"[{item.severity.value.upper()}] {item.column}: {item.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------- metrics


def population_stability_index(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    """PSI between two sets of bin shares (each sums to 1)."""
    ref = np.clip(reference_share, EPSILON, None)
    cur = np.clip(current_share, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _numeric_psi(reference: pd.Series, current: pd.Series) -> float | None:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, BINS + 1)[1:-1]))
    if len(edges) == 0:
        return None
    ref_bins = np.searchsorted(edges, reference.to_numpy(), side="right")
    cur_bins = np.searchsorted(edges, current.to_numpy(), side="right")
    size = len(edges) + 1
    ref_share = np.bincount(ref_bins, minlength=size) / len(reference)
    cur_share = np.bincount(cur_bins, minlength=size) / len(current)
    return population_stability_index(ref_share, cur_share)


def _category_psi(reference: pd.Series, current: pd.Series) -> tuple[float, list[str], list[str]]:
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    labels = ref_counts.index.union(cur_counts.index)
    ref_share = ref_counts.reindex(labels, fill_value=0).to_numpy() / len(reference)
    cur_share = cur_counts.reindex(labels, fill_value=0).to_numpy() / len(current)
    new = sorted(str(label) for label in cur_counts.index.difference(ref_counts.index))
    gone = sorted(str(label) for label in ref_counts.index.difference(cur_counts.index))
    return population_stability_index(ref_share, cur_share), new[:10], gone[:10]


def _severity_from(psi: float | None, missing_shift: float) -> Severity:
    level = Severity.INFO
    if psi is not None:
        if psi >= PSI_LARGE:
            level = Severity.HIGH
        elif psi >= PSI_MODERATE:
            level = Severity.MEDIUM
    if abs(missing_shift) >= MISSING_SHIFT_LARGE:
        level = Severity.HIGH
    elif abs(missing_shift) >= MISSING_SHIFT_MODERATE and level.rank < Severity.MEDIUM.rank:
        level = Severity.MEDIUM
    return level


# ---------------------------------------------------------------- one column


def _compare_column(
    name: str, kind: ColumnKind, ref_ctx: CheckContext, cur_ctx: CheckContext
) -> ColumnDrift:
    ref_missing = float(ref_ctx.df[name].isna().mean())
    cur_missing = float(cur_ctx.df[name].isna().mean())
    missing_shift = cur_missing - ref_missing

    fields: dict = {}
    psi: float | None = None
    parts: list[str] = []

    if kind is ColumnKind.NUMERIC:
        ref_values = ref_ctx.numeric_values(name)
        cur_values = cur_ctx.numeric_values(name)
        if len(ref_values) >= MIN_VALUES and len(cur_values) >= MIN_VALUES:
            psi = _numeric_psi(ref_values, cur_values)
            ks = stats.ks_2samp(ref_values.to_numpy(), cur_values.to_numpy())
            ref_mean, cur_mean = float(ref_values.mean()), float(cur_values.mean())
            spread = float(ref_values.std(ddof=1))
            shift = (cur_mean - ref_mean) / spread if spread > 0 else None
            fields = {
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_shift_in_std": shift,
            }
            if psi is not None:
                parts.append(f"PSI {psi:.2f}")
            parts.append(f"mean {ref_mean:.4g} -> {cur_mean:.4g}")
    elif kind in (ColumnKind.CATEGORICAL, ColumnKind.BOOLEAN, ColumnKind.CONSTANT):
        ref_values = ref_ctx.df[name].dropna().astype(str).str.strip()
        cur_values = cur_ctx.df[name].dropna().astype(str).str.strip()
        if (
            len(ref_values) >= MIN_VALUES
            and len(cur_values) >= MIN_VALUES
            and ref_values.nunique() <= MAX_CATEGORIES
        ):
            psi, new, gone = _category_psi(ref_values, cur_values)
            fields = {"new_categories": new, "gone_categories": gone}
            parts.append(f"PSI {psi:.2f}")
            if new:
                parts.append(f"new values: {', '.join(new[:5])}")
            if gone:
                parts.append(f"no longer seen: {', '.join(gone[:5])}")

    if abs(missing_shift) >= MISSING_SHIFT_MODERATE:
        parts.append(f"missing {ref_missing:.0%} -> {cur_missing:.0%}")

    severity = _severity_from(psi, missing_shift)
    message = "; ".join(parts) if parts else "not compared (too few values or not comparable)"
    if severity is Severity.INFO and parts:
        message = "stable (" + "; ".join(parts) + ")"
    return ColumnDrift(
        column=name,
        kind=kind,
        severity=severity,
        message=message,
        psi=psi,
        reference_missing=ref_missing,
        current_missing=cur_missing,
        **fields,
    )


# ---------------------------------------------------------------- public


def _load(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    table = source if isinstance(source, pd.DataFrame) else load_csv(source).df
    return prepare_frame(table)


def compare(
    reference: str | Path | pd.DataFrame,
    current: str | Path | pd.DataFrame,
    *,
    type_rules: TypeRules | None = None,
    check_rules: CheckRules | None = None,
) -> DriftReport:
    """Compare `current` against `reference`, column by column."""
    check_rules = check_rules or CheckRules()
    ref_frame, cur_frame = _load(reference), _load(current)

    def context(frame: pd.DataFrame) -> CheckContext:
        types = {item.name: item for item in infer_types(frame, type_rules)}
        return CheckContext(df=frame, types=types, rules=check_rules)

    ref_ctx, cur_ctx = context(ref_frame), context(cur_frame)

    shared = [name for name in ref_frame.columns if name in cur_frame.columns]
    removed = [name for name in ref_frame.columns if name not in cur_frame.columns]
    added = [name for name in cur_frame.columns if name not in ref_frame.columns]

    columns: list[ColumnDrift] = []
    type_changes: list[tuple[str, str, str]] = []
    for name in shared:
        ref_kind, cur_kind = ref_ctx.types[name].kind, cur_ctx.types[name].kind
        if ref_kind is not cur_kind and {ref_kind, cur_kind} != {
            ColumnKind.CATEGORICAL,
            ColumnKind.BOOLEAN,
        }:
            type_changes.append((name, ref_kind.value, cur_kind.value))
            continue
        columns.append(_compare_column(name, ref_kind, ref_ctx, cur_ctx))

    columns.sort(key=lambda item: (-item.severity.rank, -(item.psi or 0.0), item.column))
    return DriftReport(
        reference_rows=len(ref_frame),
        current_rows=len(cur_frame),
        columns=columns,
        added_columns=added,
        removed_columns=removed,
        type_changes=type_changes,
    )
