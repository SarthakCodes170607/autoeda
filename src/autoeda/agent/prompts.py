"""Prompts for the three model-driven steps. Kept in one place so they are easy to read."""

from __future__ import annotations

from autoeda.models import Report

PLANNER_SYSTEM = """You plan the analysis of a data table. You do not write code.
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
- The query must use column names exactly as written in the table schema.
- operation is one of aggregate, top_k, describe, correlation, value_counts, trend.
- aggregate: set column and agg (sum, mean, median, min, max, count, nunique) and
  optionally group_by. top_k: same, plus k and ascending. trend: set date_column, column,
  agg and freq (D, W, M, Q, Y). value_counts: set column. describe/correlation: set columns.
- filters is a list of {"column", "op", "value"}; op is one of ==, !=, >, >=, <, <=, in,
  not_in, contains, not_null.
"""

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
