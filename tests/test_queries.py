import numpy as np
import pandas as pd
import pytest

from autoeda.agent.queries import Filter, QueryError, QuerySpec, run_query


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "city": ["Delhi", "Delhi", "Pune", "Pune", "Mumbai", "Mumbai", "Mumbai"],
            "plan": ["a", "b", "a", "a", "b", "b", "a"],
            "sales": [10.0, 20.0, 5.0, 15.0, 100.0, 50.0, np.nan],
            "date": pd.to_datetime(
                [
                    "2024-01-05",
                    "2024-01-20",
                    "2024-02-03",
                    "2024-02-10",
                    "2024-03-01",
                    "2024-03-15",
                    "2024-03-20",
                ]
            ),
            "label": ["x"] * 7,
        }
    )


def run(frame, **kwargs):
    return run_query(frame, QuerySpec(**kwargs))


def test_total_without_groups_is_one_number(frame):
    result = run(frame, operation="aggregate", column="sales", agg="sum")

    assert result.table.iloc[0, 0] == 200.0
    assert result.rows_used == 7


def test_aggregate_by_group_is_sorted_largest_first(frame):
    result = run(frame, operation="aggregate", column="sales", agg="sum", group_by=["city"])

    assert result.table["city"].tolist() == ["Mumbai", "Delhi", "Pune"]
    assert result.table.iloc[0, 1] == 150.0


def test_top_k_limits_the_groups(frame):
    result = run(frame, operation="top_k", column="sales", agg="mean", group_by=["city"], k=2)

    assert result.table["city"].tolist() == ["Mumbai", "Delhi"] or len(result.table) == 2
    assert len(result.table) == 2


def test_top_k_ascending_gives_the_smallest(frame):
    result = run(
        frame, operation="top_k", column="sales", agg="sum", group_by=["city"], k=1, ascending=True
    )

    assert result.table["city"].tolist() == ["Pune"]


def test_top_k_rows_without_groups(frame):
    result = run(frame, operation="top_k", column="sales", k=2)

    assert result.table["sales"].tolist() == [100.0, 50.0]


def test_count_and_nunique_work_on_text_columns(frame):
    assert run(frame, operation="aggregate", column="city", agg="nunique").table.iloc[0, 0] == 3
    assert run(frame, operation="aggregate", column="plan", agg="count").table.iloc[0, 0] == 7


def test_filters_narrow_the_rows(frame):
    result = run(
        frame,
        operation="aggregate",
        column="sales",
        agg="sum",
        filters=[Filter(column="city", op="==", value="Pune")],
    )

    assert result.table.iloc[0, 0] == 20.0
    assert result.rows_used == 2


@pytest.mark.parametrize(
    ("op", "value", "expected_rows"),
    [
        (">", 15, 3),
        (">=", 15, 4),
        ("<", 15, 2),
        ("!=", 10, 5),
        ("in", ["Delhi", "Pune"], 4),
        ("not_in", ["Delhi"], 5),
        ("contains", "mum", 3),
        ("not_null", None, 6),
    ],
)
def test_every_filter_operator(frame, op, value, expected_rows):
    column = "sales" if op in (">", ">=", "<", "!=", "not_null") else "city"
    result = run(
        frame,
        operation="aggregate",
        column="label",
        agg="count",
        filters=[Filter(column=column, op=op, value=value)],
    )

    assert result.rows_used == expected_rows


def test_date_filter_accepts_text_dates(frame):
    result = run(
        frame,
        operation="aggregate",
        column="label",
        agg="count",
        filters=[Filter(column="date", op=">=", value="2024-02-01")],
    )

    assert result.rows_used == 5


def test_trend_groups_by_month(frame):
    result = run(frame, operation="trend", column="sales", agg="sum", date_column="date", freq="M")

    assert result.table["period"].tolist() == ["2024-01", "2024-02", "2024-03"]
    assert result.table.iloc[:, 1].tolist() == [30.0, 20.0, 150.0]


def test_value_counts(frame):
    result = run(frame, operation="value_counts", column="city", k=2)

    assert result.table["city"].tolist()[0] == "Mumbai"
    assert result.table["count"].tolist()[0] == 3
    assert len(result.table) == 2


def test_describe_and_correlation(frame):
    frame["other"] = frame["sales"] * 2
    described = run(frame, operation="describe", columns=["sales"])
    correlated = run(frame, operation="correlation", columns=["sales", "other"])

    assert described.table["count"].iloc[0] == 6
    assert correlated.table.iloc[0, 2] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"operation": "aggregate", "column": "nope"}, "no column 'nope'"),
        ({"operation": "aggregate", "column": "Sales", "agg": "sum"}, "Did you mean 'sales'"),
        ({"operation": "aggregate", "column": "city", "agg": "sum"}, "not numeric"),
        ({"operation": "aggregate", "agg": "sum"}, "column is required"),
        ({"operation": "aggregate", "column": "sales", "group_by": ["zzz"]}, "no column 'zzz'"),
        ({"operation": "trend", "column": "sales", "date_column": "city"}, "not a date"),
        ({"operation": "correlation", "columns": ["sales"]}, "at least two"),
        (
            {
                "operation": "aggregate",
                "column": "sales",
                "filters": [{"column": "city", "op": "==", "value": "Nowhere"}],
            },
            "No rows match",
        ),
        (
            {
                "operation": "aggregate",
                "column": "sales",
                "filters": [{"column": "sales", "op": ">", "value": "abc"}],
            },
            "Cannot compare",
        ),
        (
            {
                "operation": "aggregate",
                "column": "sales",
                "filters": [{"column": "sales", "op": ">"}],
            },
            "needs a value",
        ),
    ],
)
def test_invalid_queries_raise_helpful_errors(frame, spec, message):
    with pytest.raises(QueryError, match=message):
        run(frame, **spec)


def test_the_input_table_is_never_changed(frame):
    before = frame.copy()

    run(frame, operation="aggregate", column="sales", group_by=["city"])

    pd.testing.assert_frame_equal(frame, before)


def test_markdown_table_has_header_divider_and_rows(frame):
    text = run(frame, operation="aggregate", column="sales", group_by=["city"]).to_markdown()

    lines = text.splitlines()
    assert lines[0].startswith("| city")
    assert set(lines[1]) <= {"|", "-"}
    assert len(lines) == 5
