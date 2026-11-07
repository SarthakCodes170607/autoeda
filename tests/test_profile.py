from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoeda import Report, Severity, profile
from autoeda.checks import check_duplicates
from autoeda.types import ColumnKind

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_messy.csv"


@pytest.fixture(scope="module")
def sample_report():
    return profile(SAMPLE)


def test_sample_file_reports_every_planted_problem(sample_report):
    ids = {item.id for item in sample_report.issues}

    expected = {
        "empty_column:internal_flag",
        "constant_column:country",
        "duplicates:rows",
        "numeric_as_text:amount",
        "disguised_missing:amount",
        "date_as_text:order_date",
        "outliers:age",
        "outliers:unit_price",
        "missing:unit_price",
        "missing:notes",
        "id_column:order_id",
    }
    assert expected <= ids


def test_sample_file_has_no_issue_on_healthy_columns(sample_report):
    ids = {item.id for item in sample_report.issues}

    for healthy in ("city", "quantity", "is_returned"):
        assert not any(issue_id.endswith(f":{healthy}") for issue_id in ids)


def test_sample_file_details(sample_report):
    duplicates = sample_report.issue("duplicates:rows")
    date_issue = sample_report.issue("date_as_text:order_date")
    age_outlier = sample_report.issue("outliers:age")

    assert duplicates.details["count"] == 6
    assert date_issue.details["dayfirst"] is True
    assert 250.0 in age_outlier.details["examples"]


def test_issues_are_sorted_most_severe_first(sample_report):
    ranks = [item.severity.rank for item in sample_report.issues]

    assert ranks == sorted(ranks, reverse=True)
    assert sample_report.issues[0].severity is Severity.HIGH


def test_quality_score_is_between_zero_and_one_hundred(sample_report):
    assert 0 <= sample_report.quality_score < 100


def test_clean_data_scores_one_hundred():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})

    report = profile(frame)

    assert report.quality_score == 100.0


def test_report_survives_a_json_round_trip(sample_report):
    text = sample_report.model_dump_json()

    restored = Report.model_validate_json(text)

    assert restored == sample_report


def test_summary_text_lists_the_worst_issues_first(sample_report):
    text = sample_report.summary_text(max_issues=3)

    lines = text.splitlines()
    assert lines[0].startswith("Dataset: 306 rows x 12 columns")
    assert lines[2].startswith("[HIGH]")
    assert "more" in lines[-1]


def test_severity_counts_add_up_to_the_issue_count(sample_report):
    assert sum(sample_report.severity_counts().values()) == len(sample_report.issues)


# ---------------------------------------------------------------- column profiles


def test_numeric_column_profile_has_stats_and_a_histogram(sample_report):
    column = sample_report.column("age")

    assert column.kind is ColumnKind.NUMERIC
    assert column.stats["max"] == 250.0
    assert column.stats["count"] == 306
    assert sum(column.histogram["counts"]) == 306
    assert len(column.histogram["edges"]) == len(column.histogram["counts"]) + 1


def test_numbers_stored_as_text_get_numeric_stats(sample_report):
    column = sample_report.column("amount")

    assert column.stored_as_text is True
    assert 200 < column.stats["median"] < 9000


def test_dates_stored_as_text_get_a_range(sample_report):
    column = sample_report.column("order_date")

    assert column.stats["min"].startswith("2024-01")
    assert column.stats["max"].startswith("2024-12")


def test_categorical_column_profile_lists_top_values(sample_report):
    column = sample_report.column("city")

    assert len(column.top_values) == 6
    assert sum(count for _, count in column.top_values) == 306


def test_correlation_matrix_is_square_and_has_ones_on_the_diagonal(sample_report):
    matrix = sample_report.correlation

    assert matrix is not None
    assert len(matrix.values) == len(matrix.labels)
    assert all(matrix.values[i][i] == 1.0 for i in range(len(matrix.labels)))


def test_no_correlation_matrix_with_a_single_numeric_column():
    report = profile(pd.DataFrame({"a": np.arange(50) * 1.5, "b": ["x", "y"] * 25}))

    assert report.correlation is None


# ---------------------------------------------------------------- inputs


def test_dataframe_input_matches_file_input():
    from_file = profile(SAMPLE)
    from_frame = profile(pd.read_csv(SAMPLE))

    assert {item.id for item in from_frame.issues} == {item.id for item in from_file.issues}


def test_max_rows_limits_the_analysis_and_says_so():
    report = profile(SAMPLE, max_rows=100)

    assert report.n_rows == 100
    assert report.truncated is True
    assert "first 100 rows" in report.notes[0]


def test_max_rows_applies_to_dataframes_too():
    report = profile(pd.DataFrame({"a": np.arange(200) * 1.5}), max_rows=50)

    assert report.n_rows == 50
    assert report.truncated is True


def test_numeric_column_names_are_converted_to_text():
    frame = pd.DataFrame(np.random.default_rng(0).normal(size=(60, 2)), columns=[0, 1])

    report = profile(frame)

    assert [item.name for item in report.columns] == ["0", "1"]


def test_duplicate_column_names_are_rejected_with_a_clear_message():
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])

    with pytest.raises(ValueError, match="unique"):
        profile(frame)


def test_empty_tables_are_rejected():
    with pytest.raises(ValueError, match="no rows"):
        profile(pd.DataFrame({"a": []}))
    with pytest.raises(ValueError, match="no columns"):
        profile(pd.DataFrame(index=range(3)))


def test_row_labels_in_issues_are_positions_even_for_a_custom_index():
    frame = pd.DataFrame({"a": list(range(20)) + [3, 4]}, index=range(500, 522))

    report = profile(frame)

    assert report.issue("duplicates:rows").details["example_index"] == [20, 21]


def test_only_the_requested_checks_run():
    frame = pd.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})

    report = profile(frame, checks=(check_duplicates,))

    assert {item.kind.value for item in report.issues} == {"duplicates"}


def test_single_row_and_single_column_tables_do_not_crash():
    assert profile(pd.DataFrame({"a": [1.5]})).n_rows == 1
    assert profile(pd.DataFrame({"a": ["x", "y", "z"]})).n_cols == 1


def test_very_odd_tables_do_not_crash_or_warn():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "all_missing": [np.nan] * 120,
            "inf_values": np.where(rng.random(120) < 0.05, np.inf, rng.normal(size=120)),
            "mixed": [1, "two", 3.5, None] * 30,
            "bools": rng.random(120) < 0.5,
            "dates": pd.date_range("2024-01-01", periods=120),
            "long_text": ["a fairly long sentence about nothing in particular " * 3] * 120,
        }
    )

    report = profile(frame)

    assert report.n_cols == 6
