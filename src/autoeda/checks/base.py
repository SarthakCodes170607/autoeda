"""Shared helpers for the checks."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from autoeda.models import Issue, IssueKind, Severity
from autoeda.rules import CheckRules
from autoeda.types import ColumnKind, ColumnType, parse_numeric_text


@dataclass
class CheckContext:
    """The table, what we know about its columns, and caches shared between checks."""

    df: pd.DataFrame
    types: dict[str, ColumnType]
    rules: CheckRules
    _numeric_cache: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    _correlation_cache: dict[tuple[str, ...], pd.DataFrame] = field(
        default_factory=dict, repr=False
    )

    @property
    def n_rows(self) -> int:
        return len(self.df)

    def names_of_kind(self, *kinds: ColumnKind) -> list[str]:
        return [name for name, info in self.types.items() if info.kind in kinds]

    def is_text_storage(self, name: str) -> bool:
        series = self.df[name]
        return bool(pdt.is_object_dtype(series) or pdt.is_string_dtype(series))

    def numeric_values(self, name: str) -> pd.Series:
        """The column's numbers as floats, missing, infinite and unreadable values removed.

        Numbers stored as text ("$1,234.50") are read the same way the type detection read
        them. The original row labels are kept.
        """
        if name in self._numeric_cache:
            return self._numeric_cache[name]

        series = self.df[name]
        if self.types[name].stored_as_text:
            text = series.dropna().astype(str).str.strip()
            numbers = parse_numeric_text(text)
        else:
            numbers = pd.to_numeric(series, errors="coerce")

        # Infinity cannot be averaged or plotted, so it is treated like a missing value.
        result = numbers.astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
        self._numeric_cache[name] = result
        return result

    def numeric_frame(self, names: list[str]) -> pd.DataFrame:
        """Several numeric columns side by side, aligned on the table's rows."""
        columns = {name: self.numeric_values(name).reindex(self.df.index) for name in names}
        return pd.DataFrame(columns, index=self.df.index)

    def correlation(self, names: list[str]) -> pd.DataFrame:
        """Pearson correlation between numeric columns, ignoring missing values pairwise."""
        key = tuple(names)
        if key not in self._correlation_cache:
            frame = self.numeric_frame(names)
            # Near-constant columns make numpy divide by (almost) zero; the result is NaN
            # and is handled by the callers.
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                self._correlation_cache[key] = frame.corr(method="pearson")
        return self._correlation_cache[key]


Check = Callable[[CheckContext], list[Issue]]


def make_id(kind: IssueKind, *columns: str) -> str:
    return f"{kind.value}:{'|'.join(columns)}"


def plural(count: int, singular: str, plural_form: str | None = None) -> str:
    """ "1 row", "3 rows"."""
    word = singular if count == 1 else (plural_form or singular + "s")
    return f"{count} {word}"


def severity_by_fraction(
    fraction: float,
    medium: float,
    high: float,
    base: Severity = Severity.LOW,
) -> Severity:
    if fraction >= high:
        return Severity.HIGH
    if fraction >= medium:
        return Severity.MEDIUM
    return base


def prepare_text(series: pd.Series) -> pd.Series:
    """Non-missing values of a column as stripped strings; empty strings removed."""
    text = series.dropna().astype(str).str.strip()
    return text[text != ""]
