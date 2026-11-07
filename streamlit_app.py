"""Streamlit demo for autoeda: profile, clean, compare and ask questions about a CSV file."""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from autoeda import Severity, charts, clean, compare, profile
from autoeda.agent.queries import QueryError, QuerySpec, run_query
from autoeda.agent.viz import result_figure
from autoeda.fixes import fix_to_code
from autoeda.loader import load_csv
from autoeda.report import render_report

MAX_ROWS = 50_000  # the free hosting tier has about 1 GB of memory
SAMPLE = Path(__file__).parent / "examples" / "sample_messy.csv"
LEVEL_LABEL = {
    Severity.HIGH: "HIGH",
    Severity.MEDIUM: "MEDIUM",
    Severity.LOW: "LOW",
    Severity.INFO: "INFO",
}

st.set_page_config(page_title="autoeda", page_icon=None, layout="wide")


# ---------------------------------------------------------------- loading


def read_upload(upload) -> tuple[pd.DataFrame, bool]:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        handle.write(upload.getvalue())
        path = Path(handle.name)
    try:
        loaded = load_csv(path, max_rows=MAX_ROWS)
    finally:
        path.unlink(missing_ok=True)
    return loaded.df, loaded.truncated


@st.cache_data(show_spinner="Profiling...")
def cached_profile(df: pd.DataFrame):
    return profile(df)


def current_table() -> tuple[pd.DataFrame | None, str, bool]:
    with st.sidebar:
        st.header("Data")
        upload = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"])
        use_sample = st.button("Use the sample messy file", use_container_width=True)
        st.caption(f"Files are read up to {MAX_ROWS:,} rows. Nothing is stored.")

    if upload is not None:
        try:
            df, truncated = read_upload(upload)
        except Exception as error:  # a malformed file should show a message, not a stack trace
            st.sidebar.error(f"Could not read that file: {error}")
            return None, "", False
        st.session_state["table"] = (df, upload.name, truncated)
    elif use_sample or ("table" not in st.session_state and SAMPLE.exists()):
        loaded = load_csv(SAMPLE)
        st.session_state["table"] = (loaded.df, SAMPLE.name, False)

    stored = st.session_state.get("table")
    if stored is None:
        return None, "", False
    return stored


# ---------------------------------------------------------------- tabs


def profile_tab(df: pd.DataFrame, name: str, truncated: bool) -> None:
    report = cached_profile(df)
    counts = report.severity_counts()

    columns = st.columns(5)
    columns[0].metric("Quality score", f"{report.quality_score:.0f} / 100")
    columns[1].metric("Rows", f"{report.n_rows:,}")
    columns[2].metric("Columns", report.n_cols)
    columns[3].metric("High", counts["high"])
    columns[4].metric("Medium", counts["medium"])
    if truncated:
        st.info(f"Only the first {MAX_ROWS:,} rows were analysed.")

    st.subheader(f"Issues ({len(report.issues)})")
    if not report.issues:
        st.success("No issues found.")
    for issue in report.issues:
        label = f"{LEVEL_LABEL[issue.severity]}  ·  {issue.title}"
        if issue.column:
            label += f"  ·  {issue.column}"
        with st.expander(label):
            st.write(issue.message)
            if issue.fix:
                tag = " (safe to apply automatically)" if issue.fix.safe_auto else ""
                st.caption(f"Suggested fix: {issue.fix.description}{tag}")
                st.code(fix_to_code(issue.fix), language="python")

    left, right = st.columns(2)
    missing = charts.missing_values(report)
    if missing is not None:
        left.subheader("Missing values")
        left.plotly_chart(missing, use_container_width=True)
    correlation = charts.correlation(report.correlation)
    if correlation is not None:
        right.subheader("Correlation")
        right.plotly_chart(correlation, use_container_width=True)

    st.subheader("Columns")
    chosen = st.selectbox("Look at a column", [item.name for item in report.columns])
    column = report.column(chosen)
    st.caption(f"{column.kind.value}: {column.reason}")
    figure = charts.column_chart(column)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    if column.stats:
        st.json(column.stats, expanded=False)

    st.download_button(
        "Download HTML report",
        render_report(report, plotly_js="inline"),
        file_name="autoeda_report.html",
        mime="text/html",
    )


def clean_tab(df: pd.DataFrame) -> None:
    report = cached_profile(df)
    fixable = [issue.id for issue in report.issues if issue.fix is not None]
    mode = st.radio(
        "Which fixes?",
        ["Safe fixes only", "Every suggestion", "Let me choose"],
        horizontal=True,
        help=(
            "Safe fixes remove defects without judgment (stripping spaces, reading numbers and "
            "dates, merging spellings, dropping duplicates). The other suggestions, such as "
            "filling missing values or clipping outliers, change what the data says."
        ),
    )
    chosen: list[str] | None = None
    if mode == "Let me choose":
        chosen = st.multiselect("Issues to fix", fixable)

    if st.button("Clean the data", type="primary"):
        if mode == "Safe fixes only":
            result = clean(df, report, fixes="safe")
        elif mode == "Every suggestion":
            result = clean(df, report, fixes="all")
        else:
            result = clean(df, report, fixes=chosen or [])
        st.session_state["cleaned"] = result

    result = st.session_state.get("cleaned")
    if result is None:
        return

    left, right = st.columns(2)
    left.metric(
        "Quality score",
        f"{result.after.quality_score:.0f}",
        f"{result.after.quality_score - result.before.quality_score:+.0f}",
    )
    right.metric(
        "Issues",
        len(result.after.issues),
        len(result.after.issues) - len(result.before.issues),
        delta_color="inverse",
    )

    rows = [
        {"issue": item.issue_id, "what was done": item.fix.description, "effect": item.effect}
        for item in result.applied
    ] + [
        {"issue": item.issue_id, "what was done": "skipped", "effect": item.reason}
        for item in result.skipped
    ]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.subheader("The same steps as pandas code")
    st.code(result.code, language="python")
    st.subheader("Cleaned data (first rows)")
    st.dataframe(result.df.head(50), use_container_width=True)

    left, right = st.columns(2)
    left.download_button(
        "Download cleaned CSV",
        result.df.to_csv(index=False),
        file_name="cleaned.csv",
        mime="text/csv",
    )
    right.download_button("Download script", result.code, file_name="clean.py")


def compare_tab(df: pd.DataFrame) -> None:
    st.write(
        "Upload a second file to see how it differs from the current one: shifted averages, "
        "new categories, more missing values, added or removed columns."
    )
    other = st.file_uploader("Newer CSV file", type=["csv", "tsv", "txt"], key="compare_upload")
    if other is None:
        return
    newer, _ = read_upload(other)
    drift = compare(df, newer)

    st.text(drift.summary_text())
    figure = charts.drift_scores(drift)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    rows = [
        {
            "column": item.column,
            "severity": LEVEL_LABEL[item.severity],
            "what changed": item.message,
        }
        for item in drift.columns
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def ask_tab(df: pd.DataFrame) -> None:
    result = st.session_state.get("cleaned")
    if result is None:
        result = clean(df, fixes="safe")
        st.session_state["cleaned"] = result

    st.subheader("Ask in plain English")
    st.caption(
        "A language model chooses the fixes and the query; it never writes code, and numbers in "
        "its answer are checked against the result. This needs an Ollama server, so it works "
        "when you run the app on your own computer."
    )
    question = st.text_input("Question", placeholder="What are the total sales for each region?")
    left, right = st.columns(2)
    model = left.text_input("Ollama model", value="qwen2.5:7b-instruct")
    host = right.text_input("Ollama address", value="http://localhost:11434")
    if st.button("Ask") and question:
        from autoeda.agent import OllamaClient, ask

        with st.spinner("Thinking..."):
            outcome = ask(df, question, OllamaClient(model=model, host=host))
        st.write(outcome.answer)
        if outcome.result is not None:
            st.dataframe(outcome.result.table, use_container_width=True, hide_index=True)
        if outcome.figure is not None:
            st.plotly_chart(outcome.figure, use_container_width=True)
        with st.expander("What the agent did"):
            for step in outcome.trace:
                st.write(f"**{step['step']}**: {step['detail']}")

    st.divider()
    st.subheader("Or build a query yourself (no model needed)")
    table = result.df
    numeric = [c for c in table.columns if pd.api.types.is_numeric_dtype(table[c])]
    operation = st.selectbox("Operation", ["aggregate", "top_k", "value_counts", "trend"])
    group_by = st.multiselect("Group by", list(table.columns))
    column = st.selectbox("Column", list(table.columns), index=0)
    agg = st.selectbox("Aggregation", ["sum", "mean", "median", "min", "max", "count", "nunique"])
    dates = [c for c in table.columns if pd.api.types.is_datetime64_any_dtype(table[c])]
    date_column = st.selectbox("Date column (for trend)", dates) if dates else None
    k = st.slider("Rows to show (top_k, value_counts)", 1, 20, 5)
    if st.button("Run query"):
        spec = QuerySpec(
            operation=operation,
            column=column,
            agg=agg,
            group_by=group_by,
            k=k,
            date_column=date_column,
        )
        try:
            output = run_query(table, spec)
        except QueryError as error:
            st.error(str(error))
        else:
            st.caption(f"{output.title}, from {output.rows_used:,} rows")
            st.dataframe(output.table, use_container_width=True, hide_index=True)
            figure = result_figure(output)
            if figure is not None:
                st.plotly_chart(figure, use_container_width=True)
    if not numeric:
        st.caption("This table has no numeric columns, so only counts are available.")


# ---------------------------------------------------------------- page


def main() -> None:
    st.title("autoeda")
    st.caption("Find data quality problems, fix them, compare versions, and ask questions.")

    df, name, truncated = current_table()
    if df is None:
        st.info("Upload a CSV file in the sidebar to begin.")
        return
    if st.session_state.get("table_name") != name:
        st.session_state["table_name"] = name
        st.session_state.pop("cleaned", None)

    st.write(f"**{name}**: {len(df):,} rows, {df.shape[1]} columns")
    profile_view, clean_view, compare_view, ask_view = st.tabs(
        ["Profile", "Clean", "Compare", "Ask"]
    )
    with profile_view:
        profile_tab(df, name, truncated)
    with clean_view:
        clean_tab(df)
    with compare_view:
        compare_tab(df)
    with ask_view:
        ask_tab(df)


main()
