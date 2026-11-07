"""Data models shared by every part of autoeda: issues, fixes, column profiles, reports."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from autoeda.types import ColumnKind


class Severity(str, Enum):
    """How much an issue matters. INFO is a remark, HIGH usually breaks analysis."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        return SEVERITY_RANK[self]


SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
}


class IssueKind(str, Enum):
    MISSING_VALUES = "missing"
    DISGUISED_MISSING = "disguised_missing"
    EMPTY_COLUMN = "empty_column"
    CONSTANT_COLUMN = "constant_column"
    DUPLICATE_ROWS = "duplicates"
    OUTLIERS = "outliers"
    MULTIVARIATE_OUTLIERS = "multivariate_outliers"
    SENTINEL_VALUES = "sentinels"
    NUMERIC_AS_TEXT = "numeric_as_text"
    DATE_AS_TEXT = "date_as_text"
    WHITESPACE = "whitespace"
    CASE_VARIANTS = "case_variants"
    HIGH_CORRELATION = "correlated"
    SKEWED = "skewed"
    CLASS_IMBALANCE = "imbalance"
    HIGH_CARDINALITY = "high_cardinality"
    ID_COLUMN = "id_column"
    MISSING_TOGETHER = "missing_together"


class FixAction(str, Enum):
    DROP_COLUMN = "drop_column"
    DROP_DUPLICATE_ROWS = "drop_duplicate_rows"
    STRIP_WHITESPACE = "strip_whitespace"
    REPLACE_WITH_NAN = "replace_with_nan"
    CONVERT_NUMERIC = "convert_numeric"
    CONVERT_DATETIME = "convert_datetime"
    NORMALIZE_CASE = "normalize_case"
    IMPUTE_MEDIAN = "impute_median"
    IMPUTE_MODE = "impute_mode"
    CLIP_OUTLIERS = "clip_outliers"
    LOG_TRANSFORM = "log_transform"


class Fix(BaseModel):
    """A concrete, machine-applicable repair for an issue."""

    action: FixAction
    column: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    # True when the repair only removes a defect and needs no judgment from the user
    # (stripping spaces, converting "1,234" to 1234). Modelling decisions such as filling
    # missing values or clipping outliers are never automatic.
    safe_auto: bool = False
    description: str


class Issue(BaseModel):
    """One problem (or remark) found in the data."""

    id: str
    kind: IssueKind
    severity: Severity
    column: str | None = None
    columns: list[str] = Field(default_factory=list)
    title: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    fix: Fix | None = None

    @property
    def affected_fraction(self) -> float:
        return float(self.details.get("fraction", 0.0))


class ColumnProfile(BaseModel):
    """Everything we know about one column."""

    name: str
    kind: ColumnKind
    dtype: str
    n_rows: int
    n_missing: int
    missing_fraction: float
    n_unique: int
    stored_as_text: bool
    reason: str
    stats: dict[str, Any] = Field(default_factory=dict)
    top_values: list[tuple[str, int]] = Field(default_factory=list)
    histogram: dict[str, list[float]] | None = None


class CorrelationMatrix(BaseModel):
    labels: list[str]
    values: list[list[float | None]]


class Report(BaseModel):
    """The result of profiling a dataset."""

    source: str | None = None
    n_rows: int
    n_cols: int
    truncated: bool = False
    columns: list[ColumnProfile]
    issues: list[Issue]
    quality_score: float
    correlation: CorrelationMatrix | None = None
    notes: list[str] = Field(default_factory=list)

    def issue(self, issue_id: str) -> Issue | None:
        for item in self.issues:
            if item.id == issue_id:
                return item
        return None

    def column(self, name: str) -> ColumnProfile | None:
        for item in self.columns:
            if item.name == name:
                return item
        return None

    def issues_at_least(self, severity: Severity) -> list[Issue]:
        return [item for item in self.issues if item.severity.rank >= severity.rank]

    def severity_counts(self) -> dict[str, int]:
        counts = {level.value: 0 for level in Severity}
        for item in self.issues:
            counts[item.severity.value] += 1
        return counts

    def summary_text(self, max_issues: int = 15, min_severity: Severity = Severity.LOW) -> str:
        """A compact plain-text summary, used by the CLI and in prompts."""
        counts = self.severity_counts()
        lines = [
            f"Dataset: {self.n_rows} rows x {self.n_cols} columns"
            + (" (file was cut short)" if self.truncated else "")
            + f". Quality score {self.quality_score:.0f}/100.",
            "Issues: "
            + ", ".join(f"{counts[level.value]} {level.value}" for level in reversed(Severity)),
        ]
        shown = self.issues_at_least(min_severity)
        for item in shown[:max_issues]:
            lines.append(f"[{item.severity.value.upper()}] {item.id}: {item.message}")
        if len(shown) > max_issues:
            lines.append(f"... and {len(shown) - max_issues} more")
        return "\n".join(lines)
