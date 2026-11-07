"""Checks that keep the written answer honest.

The writer model sees a table of facts and must describe them. These helpers verify that
every number in its answer actually comes from those facts, and build a plain answer
without any model when it does not.
"""

from __future__ import annotations

import re

import pandas as pd

from autoeda.agent.queries import QueryResult

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")
SMALL_WORDS_MAX = 10  # "top 3", "2 columns": counting words are not data claims


def _parse(token: str) -> float | None:
    cleaned = token.rstrip("%").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def numbers_in(text: str) -> list[float]:
    values = []
    for token in NUMBER.findall(text):
        value = _parse(token)
        if value is not None:
            values.append(value)
    return values


def fact_numbers(result: QueryResult, extra_text: str = "") -> set[float]:
    """Every number the writer is allowed to mention."""
    facts: set[float] = {float(result.rows_used), float(len(result.table))}
    for column in result.table.columns:
        series = result.table[column]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            facts.update(float(value) for value in series.dropna())
    for column in result.table.select_dtypes(exclude="number").columns:
        for cell in result.table[column].astype(str):
            facts.update(numbers_in(cell))
    facts.update(numbers_in(extra_text))
    return facts


def _matches(value: float, facts: set[float]) -> bool:
    """True if `value` is a fact, possibly rounded, as a percentage, or written in thousands."""
    for fact in facts:
        for candidate in (fact, fact * 100, fact / 1000, fact / 1_000_000):
            if abs(value - candidate) <= max(abs(candidate) * 0.005, 0.005):
                return True
            for digits in (0, 1, 2):
                if abs(value - round(candidate, digits)) < 1e-9:
                    return True
    return False


def ungrounded_numbers(answer: str, result: QueryResult, *, allowed_text: str = "") -> list[float]:
    """Numbers in the answer that do not appear in the facts (or the question)."""
    facts = fact_numbers(result, allowed_text)
    allowed = set(numbers_in(allowed_text))
    bad = []
    for value in numbers_in(answer):
        if value in allowed:
            continue
        if float(value).is_integer() and 0 <= value <= SMALL_WORDS_MAX:
            continue
        if not _matches(value, facts):
            bad.append(value)
    return bad


def plain_answer(question: str, result: QueryResult) -> str:
    """A model-free answer: the title and the first rows of the result."""
    table = result.table
    if len(table) == 1 and len(table.columns) == 1:
        value = table.iloc[0, 0]
        shown = f"{value:,.2f}" if isinstance(value, float) else str(value)
        return f"The {result.title} is {shown} (based on {result.rows_used:,} rows)."
    return (
        f"{result.title.capitalize()} (based on {result.rows_used:,} rows):\n\n"
        + result.to_markdown(max_rows=10)
    )
