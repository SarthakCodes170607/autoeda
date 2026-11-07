"""Work out what kind of data each column holds.

pandas gives every column a dtype (int64, object, ...), but a dtype is only a storage
format. For analysis we need the meaning: is this column a measurement, a category,
a date, an identifier? This module decides that with pandas and numpy functions plus a
small set of documented rules. Every decision carries a human-readable reason.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from pandas.api import types as pdt


class ColumnKind(str, Enum):
    """The meaning of a column, as far as we can tell."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    TEXT = "text"
    ID = "id"
    CONSTANT = "constant"
    EMPTY = "empty"
    OTHER = "other"


@dataclass(frozen=True)
class TypeRules:
    """Thresholds used by the type rules. Defaults are sensible; all can be changed."""

    # How many values we test when checking whether text looks like dates.
    sample_size: int = 1000
    # Share of values that must parse as numbers/dates for text to count as such.
    parse_ratio_threshold: float = 0.90
    # A column with at most this many distinct values may be categorical.
    categorical_max_unique: int = 20
    # ...and, for integers, distinct values must also be a small share of the rows.
    numeric_categorical_max_ratio: float = 0.05
    # For text: up to this share of distinct values still counts as categorical.
    string_categorical_max_ratio: float = 0.50
    # Share of distinct values at which a column is "unique in practice".
    id_min_unique_ratio: float = 0.99
    # When the column *name* says it is an identifier (order_id), we trust that and accept
    # a lower share, because real ID columns often contain a few duplicated rows.
    id_name_min_unique_ratio: float = 0.90
    # Minimum rows before "all values unique and same length" is trusted as an ID.
    id_min_rows: int = 20
    # Text with at least this many words per value on average is free text.
    text_min_avg_words: float = 3.0
    # If more than this share of values start with a zero (like 00123), they are codes.
    leading_zero_share: float = 0.50


@dataclass(frozen=True)
class ColumnType:
    """What we concluded about one column, and why."""

    name: str
    kind: ColumnKind
    dtype: str  # the pandas storage type, for reference
    n_rows: int
    n_missing: int  # real missing values only (NaN/None)
    n_unique: int  # distinct non-missing values
    stored_as_text: bool  # numbers or dates that arrived as strings
    parse_ratio: float | None  # share of text values that parsed, when stored as text
    reason: str


@dataclass(frozen=True)
class _Verdict:
    """Internal result of a classification rule, before it becomes a ColumnType."""

    kind: ColumnKind
    reason: str
    stored_as_text: bool = False
    parse_ratio: float | None = None


# Pairs of words that mean true/false. A two-value text column using one pair is boolean.
BOOLEAN_WORD_PAIRS = (
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"t", "f"}),
    frozenset({"0", "1"}),
)

# Words in a column name that suggest an identifier.
ID_NAME_TOKENS = frozenset({"id", "uuid", "guid", "key", "idx", "index", "sku"})

# Text that really means "no value". Compared after stripping and lower-casing. An empty
# string is handled separately because it is removed before these are compared.
DISGUISED_MISSING_TOKENS = frozenset(
    {
        "?",
        "-",
        "--",
        "---",
        ".",
        "n/a",
        "na",
        "n.a.",
        "nan",
        "null",
        "none",
        "nil",
        "missing",
        "unknown",
        "not available",
        "not applicable",
        "#n/a",
        "undefined",
        "tbd",
    }
)

# Characters removed before trying to read text as a number: commas, spaces,
# currency symbols and percent signs ("$1,234.50" becomes "1234.50").
NUMBER_NOISE_PATTERN = r"[,\s$€£₹%]"

# Text starting with a zero followed by digits, like zip codes "01234".
LEADING_ZERO_PATTERN = r"^0\d+"

# Three shapes of date-like text. Only non-capturing groups are used on purpose:
# pandas warns when a pattern given to str.contains has capture groups.
DATE_LIKE_PATTERN = (
    r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}"  # 2024-03-15, 15/03/2024, 15.03.2024
    r"|^\d{1,2}:\d{2}"  # 14:30
    r"|^(?=.*\d)(?=.*\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)"  # 15 Mar 2024
)


def _looks_like_id_name(name: object) -> bool:
    """True for names like id, customer_id, customerId, Unnamed: 0."""
    text = str(name)

    # pandas calls a header-less first column "Unnamed: 0". It is nearly always a row index.
    if text.startswith("Unnamed"):
        return True

    # Split camelCase ("customerId" -> "customer Id"), then split on anything not a letter/digit.
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    tokens = re.split(r"[^a-z0-9]+", spaced.lower())

    return any(token in ID_NAME_TOKENS for token in tokens)


def _clean_text(values: pd.Series) -> pd.Series:
    """Turn values into stripped strings and drop the ones that are empty."""
    as_text = values.astype(str).str.strip()
    return as_text[as_text != ""]


def parse_numeric_text(values: pd.Series) -> pd.Series:
    """Try to read text as numbers. Values that are not numbers become NaN."""
    cleaned = values.str.replace(NUMBER_NOISE_PATTERN, "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _datetime_parse_ratio(sample: pd.Series) -> float:
    """Share of the sample that pandas can read as a date or time."""
    # Day-first versus month-first is ambiguous for text like 03/04/2024, and pandas
    # warns about it. For *detecting* dates the ambiguity does not matter (both readings
    # succeed), so the warning is silenced here and only here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return float(parsed.notna().mean())


def _classify_numeric(values: pd.Series, name: object, rules: TypeRules) -> _Verdict:
    """Decide among boolean, id, categorical and numeric for non-missing numbers."""
    numbers = values.astype("float64").to_numpy()
    n_valid = len(numbers)
    unique_numbers = np.unique(numbers)
    n_unique = len(unique_numbers)
    unique_ratio = n_unique / n_valid

    if n_unique == 2 and unique_numbers.tolist() == [0.0, 1.0]:
        return _Verdict(ColumnKind.BOOLEAN, "only the values 0 and 1")

    # A number is whole if rounding down changes nothing (this also keeps inf from warning).
    is_integer_valued = bool(np.all(numbers == np.floor(numbers)))

    if (
        is_integer_valued
        and unique_ratio >= rules.id_name_min_unique_ratio
        and _looks_like_id_name(name)
    ):
        return _Verdict(
            ColumnKind.ID,
            f"whole numbers, {unique_ratio:.0%} distinct, and the column name looks like "
            "an identifier",
        )

    if (
        is_integer_valued
        and n_unique <= rules.categorical_max_unique
        and unique_ratio <= rules.numeric_categorical_max_ratio
    ):
        return _Verdict(
            ColumnKind.CATEGORICAL,
            f"whole numbers with only {n_unique} distinct values, so they act as category codes",
        )

    shape = "whole numbers" if is_integer_valued else "decimal numbers"
    return _Verdict(ColumnKind.NUMERIC, f"{shape} with {n_unique} distinct values")


def _classify_text(values: pd.Series, name: object, rules: TypeRules) -> _Verdict:
    """Decide what a column of strings really is. Values are already stripped, never empty."""
    # Markers such as "?" or "unknown" mean "no value". Ignore them when deciding the kind,
    # otherwise 15% of "?" would stop a numeric column from being recognised as numeric.
    is_marker = values.str.lower().isin(DISGUISED_MISSING_TOKENS)
    if bool(is_marker.all()):
        return _Verdict(
            ColumnKind.EMPTY,
            "every value is a missing-value marker such as '?' or 'unknown'",
        )
    values = values[~is_marker]
    n_valid = len(values)
    n_unique = int(values.nunique())

    # Rule 1: two values that are boolean words, ignoring upper/lower case. "Yes", "yes" and
    # "YES" are three distinct raw values but one word, so the raw count is only used as a
    # cheap filter: a column with many distinct values is not a yes/no column.
    if n_unique <= rules.categorical_max_unique:
        lowered = set(values.str.lower().unique().tolist())
        if len(lowered) == 2 and any(lowered <= pair for pair in BOOLEAN_WORD_PAIRS):
            return _Verdict(
                ColumnKind.BOOLEAN,
                f"two distinct values {sorted(lowered)} that are boolean words",
            )

    # Rule 2: numbers stored as text. Skipped for codes such as 01234, where
    # reading them as numbers would silently destroy the leading zero.
    zero_share = float(values.str.match(LEADING_ZERO_PATTERN).mean())
    if zero_share <= rules.leading_zero_share:
        numbers = parse_numeric_text(values)
        parse_ratio = float(numbers.notna().mean())
        if parse_ratio >= rules.parse_ratio_threshold:
            inner = _classify_numeric(numbers.dropna(), name, rules)
            return _Verdict(
                inner.kind,
                f"{inner.reason}; stored as text, {parse_ratio:.0%} of values parse as numbers",
                stored_as_text=True,
                parse_ratio=parse_ratio,
            )

    # Rule 3: dates stored as text. First check the text has a date-like shape,
    # because parsing is slow and too forgiving.
    if n_valid <= rules.sample_size:
        sample = values
    else:
        sample = values.sample(n=rules.sample_size, random_state=0)

    date_like_share = float(sample.str.contains(DATE_LIKE_PATTERN, case=False, regex=True).mean())
    if date_like_share >= rules.parse_ratio_threshold:
        parse_ratio = _datetime_parse_ratio(sample)
        if parse_ratio >= rules.parse_ratio_threshold:
            return _Verdict(
                ColumnKind.DATETIME,
                f"stored as text, {parse_ratio:.0%} of sampled values parse as dates",
                stored_as_text=True,
                parse_ratio=parse_ratio,
            )

    # Rule 4: identifiers, free text, or categories.
    unique_ratio = n_unique / n_valid

    if unique_ratio >= rules.id_name_min_unique_ratio and _looks_like_id_name(name):
        return _Verdict(
            ColumnKind.ID,
            f"{unique_ratio:.0%} of values are different and the column name looks like "
            "an identifier",
        )

    if unique_ratio >= rules.id_min_unique_ratio:
        same_length = int(values.str.len().nunique()) == 1
        has_spaces = bool(values.str.contains(r"\s").any())
        if n_valid >= rules.id_min_rows and same_length and not has_spaces:
            return _Verdict(
                ColumnKind.ID,
                "every value is different, all have the same length, and none contain spaces",
            )

        return _Verdict(
            ColumnKind.TEXT,
            f"almost every value is different ({n_unique} distinct in {n_valid})",
        )

    average_words = float(values.str.split().str.len().mean())
    if average_words >= rules.text_min_avg_words:
        return _Verdict(ColumnKind.TEXT, f"values average {average_words:.1f} words each")

    if unique_ratio <= rules.string_categorical_max_ratio:
        return _Verdict(
            ColumnKind.CATEGORICAL,
            f"{n_unique} distinct values across {n_valid} rows ({unique_ratio:.0%} distinct)",
        )

    return _Verdict(
        ColumnKind.TEXT,
        f"most values are different ({unique_ratio:.0%} distinct) and they are not categories",
    )


def infer_column_type(
    series: pd.Series,
    name: object,
    rules: TypeRules | None = None,
) -> ColumnType:
    """Decide the kind of a single column."""
    rules = rules if rules is not None else TypeRules()

    n_rows = len(series)
    n_missing = int(series.isna().sum())
    valid = series.dropna()

    def build(verdict: _Verdict, n_unique: int) -> ColumnType:
        return ColumnType(
            name=str(name),
            kind=verdict.kind,
            dtype=str(series.dtype),
            n_rows=n_rows,
            n_missing=n_missing,
            n_unique=n_unique,
            stored_as_text=verdict.stored_as_text,
            parse_ratio=verdict.parse_ratio,
            reason=verdict.reason,
        )

    if len(valid) == 0:
        return build(_Verdict(ColumnKind.EMPTY, "every value is missing"), 0)

    is_text_storage = pdt.is_object_dtype(series) or pdt.is_string_dtype(series)

    if is_text_storage:
        # Compare stripped strings, and treat blank strings as having no content.
        valid = _clean_text(valid)
        if len(valid) == 0:
            return build(_Verdict(ColumnKind.EMPTY, "every value is empty or whitespace"), 0)

    n_unique = int(valid.nunique())

    # One distinct value everywhere. Needs two rows or more, otherwise every column of a
    # one-row table would be reported as constant.
    if len(valid) >= 2 and n_unique == 1:
        only_value = valid.iloc[0]
        return build(
            _Verdict(ColumnKind.CONSTANT, f"every non-missing value is {only_value!r}"),
            n_unique,
        )

    if pdt.is_bool_dtype(series):
        return build(_Verdict(ColumnKind.BOOLEAN, "pandas boolean type"), n_unique)

    if pdt.is_datetime64_any_dtype(series):
        return build(_Verdict(ColumnKind.DATETIME, "pandas datetime type"), n_unique)

    if isinstance(series.dtype, pd.CategoricalDtype):
        return build(_Verdict(ColumnKind.CATEGORICAL, "pandas category type"), n_unique)

    if pdt.is_numeric_dtype(series):
        return build(_classify_numeric(valid, name, rules), n_unique)

    if is_text_storage:
        return build(_classify_text(valid, name, rules), n_unique)

    return build(_Verdict(ColumnKind.OTHER, f"unsupported pandas type {series.dtype}"), n_unique)


def infer_types(df: pd.DataFrame, rules: TypeRules | None = None) -> list[ColumnType]:
    """Decide the kind of every column, in column order."""
    column_types = []

    # Go by position, not by name, because pandas allows two columns to share a name.
    for position, name in enumerate(df.columns):
        series = df.iloc[:, position]
        column_types.append(infer_column_type(series, name, rules))

    return column_types


def types_table(column_types: list[ColumnType]) -> pd.DataFrame:
    """Show the results as a table, handy for printing and for reports."""
    rows = []
    for item in column_types:
        rows.append(
            {
                "column": item.name,
                "kind": item.kind.value,
                "dtype": item.dtype,
                "missing": item.n_missing,
                "unique": item.n_unique,
                "as_text": item.stored_as_text,
                "reason": item.reason,
            }
        )
    return pd.DataFrame(rows)
