"""Prompts for the three model-driven steps. Kept in one place so they are easy to read."""

from __future__ import annotations

import json

from autoeda.models import Report

# Worked examples for the planner. They use a made-up HR table on purpose, so they teach the
# query format without hinting at the columns or questions of any particular data set.
PLANNER_EXAMPLES: list[tuple[str, dict]] = [
    (
        "What is the average salary in each department?",
        {"operation": "aggregate", "column": "salary", "agg": "mean", "group_by": ["department"]},
    ),
    (
        "Which 3 cities have the highest total salary?",
        {
            "operation": "top_k",
            "column": "salary",
            "agg": "sum",
            "group_by": ["city"],
            "k": 3,
        },
    ),
    (
        "How many employees work remotely?",
        {
            "operation": "aggregate",
            "column": "employee_id",
            "agg": "count",
            "filters": [{"column": "remote", "op": "==", "value": "yes"}],
        },
    ),
    (
        "How many different cities do employees live in?",
        {"operation": "aggregate", "column": "city", "agg": "nunique"},
    ),
    (
        "Show the salary cost per quarter.",
        {
            "operation": "trend",
            "column": "salary",
            "agg": "sum",
            "date_column": "hired_on",
            "freq": "Q",
        },
    ),
    (
        "How many employees are in each department?",
        {"operation": "value_counts", "column": "department"},
    ),
]


def _examples_text() -> str:
    lines = []
    for question, query in PLANNER_EXAMPLES:
        lines.append(f"Question: {question}\n{json.dumps(query)}")
    return "\n\n".join(lines)


PLANNER_SYSTEM = (
    """You plan the analysis of a data table. You do not write code.
Reply with one JSON object and nothing else, in this shape:
{
  "fix_ids": [list of issue ids to repair before analysing],
  "query": {the question as a structured query},
  "rationale": "one sentence"
}

Rules:
- fix_ids may only contain ids from the list of available fixes. Choose fixes that make the
  answer wrong if left alone (numbers stored as text, placeholder values, duplicate rows,
  dates stored as text, impossible values). Do not choose fixes that change the data's
  meaning without need.
- Impossible values: if the schema shows a range that cannot be real for a column the
  question uses (an age of -999, a negative quantity), choose that column's sentinels or
  outliers fix. Outlier and skew fixes change totals and averages, so do not choose them for
  columns whose values look valid.
- The query must use column names exactly as written in the table schema.
- operation is one of aggregate, top_k, describe, correlation, value_counts, trend.
- column is the NUMBER being summed, averaged or counted (for example the amount or the id).
  group_by lists the CATEGORIES to split by. Never put the category in column.
- aggregate: set column and agg (sum, mean, median, min, max, count, nunique) and
  optionally group_by and filters. "How many X" is count of an id column (with a filter for
  a subset); "how many different X" is nunique of X; "how many items sold" is the sum of the
  quantity column, not a count of rows.
- top_k ranks GROUPS: set group_by to what you rank, plus column, agg and k (ascending for
  the smallest). Without group_by it ranks individual rows.
- trend: set date_column, column, agg and freq (D, W, M, Q, Y); column is required.
- value_counts: set column; use it only to list how often each value occurs.
- describe and correlation: set columns.
- filters is a list of {"column", "op", "value"}; op is one of ==, !=, >, >=, <, <=, in,
  not_in, contains, not_null. Use filters to restrict rows ("in the north", "returned
  orders"), not group_by.

Examples (from an unrelated HR table; the query part only):

"""
    + _examples_text()
    + "\n"
)

CRITIC_SYSTEM = """You check whether a query result answers a question.
Reply with one JSON object: {"answers_question": true or false, "concern": "short reason"}.
Say false only for a real mismatch: the wrong column or aggregation was used, a needed filter
is missing, or the result is empty or obviously nonsensical for the question."""

WRITER_SYSTEM = """You write a short answer to a question about a data table.
Use ONLY the numbers in the result table and the notes provided. Do not calculate new
numbers, and do not invent facts. Two to four sentences. Mention one caveat if the notes list
one. Plain language, no markdown headings."""


def schema_text(report: Report) -> str:
    lines = [f"Table with {report.n_rows} rows. Columns (name: kind, example/notes):"]
    for column in report.columns:
        extra = ""
        if column.top_values:
            extra = " e.g. " + ", ".join(label for label, _ in column.top_values[:4])
        elif "min" in column.stats and "max" in column.stats:
            extra = f" range {column.stats['min']} to {column.stats['max']}"
        stored = (
            " (stored as text)" if column.stored_as_text and column.kind.value != "text" else ""
        )
        lines.append(f"- {column.name}: {column.kind.value}{stored}{extra}")
    return "\n".join(lines)


def fixes_text(report: Report) -> str:
    rows = []
    for issue in report.issues:
        if issue.fix is None:
            continue
        safe = " [safe]" if issue.fix.safe_auto else ""
        rows.append(f"- {issue.id}{safe}: {issue.fix.description}")
    return "\n".join(rows) if rows else "(none)"


def planner_user(question: str, report: Report, feedback: str | None) -> str:
    parts = [
        schema_text(report),
        "\nAvailable fixes:\n" + fixes_text(report),
        f"\nQuestion: {question}",
    ]
    if feedback:
        parts.append(f"\nYour previous plan failed: {feedback}\nPlan again, correcting this.")
    return "\n".join(parts)


def critic_user(question: str, spec_json: str, result_markdown: str, rows_used: int) -> str:
    return (
        f"Question: {question}\nQuery: {spec_json}\nRows used: {rows_used}\n"
        f"Result:\n{result_markdown}"
    )


def writer_user(question: str, title: str, result_markdown: str, notes: list[str]) -> str:
    notes_text = "\n".join(f"- {note}" for note in notes) or "- none"
    return f"Question: {question}\nResult ({title}):\n{result_markdown}\nNotes:\n{notes_text}"
