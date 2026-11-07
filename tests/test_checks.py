"""Tests for the individual checks. Each test builds a small table with one known problem."""

import numpy as np
import pandas as pd
import pytest

from autoeda.checks.categorical import guess_dayfirst
from autoeda.models import FixAction, IssueKind, Severity
from autoeda.profile import profile


def issues_of(df, kind, **kwargs):
    report = profile(df, **kwargs)
    return [item for item in report.issues if item.kind is kind]


def issue_for(df, kind, column=None):
    found = [item for item in issues_of(df, kind) if column is None or item.column == column]
    assert len(found) == 1, f"expected exactly one {kind} issue, got {len(found)}"
    return found[0]


def normal_values(n=500, seed=0, mean=50.0, std=10.0):
    return np.random.default_rng(seed).normal(mean, std, n)


# ---------------------------------------------------------------- missing values


def test_missing_numeric_gets_medium_severity_and_median_fix():
    values = normal_values(200)
    values[:30] = np.nan  # 15%

    issue = issue_for(pd.DataFrame({"x": values}), IssueKind.MISSING_VALUES, "x")

    assert issue.severity is Severity.MEDIUM
    assert issue.details["count"] == 30
    assert issue.fix.action is FixAction.IMPUTE_MEDIAN
    assert issue.fix.safe_auto is False


def test_mostly_missing_column_is_high_and_suggests_dropping():
    values = normal_values(100)
    values[:70] = np.nan

    issue = issue_for(pd.DataFrame({"x": values}), IssueKind.MISSING_VALUES, "x")

    assert issue.severity is Severity.HIGH
    assert issue.fix.action is FixAction.DROP_COLUMN


def test_missing_category_gets_mode_fix():
    cities = ["Delhi", "Pune", "Delhi", None] * 25

    issue = issue_for(pd.DataFrame({"city": cities}), IssueKind.MISSING_VALUES, "city")

    assert issue.fix.action is FixAction.IMPUTE_MODE


def test_empty_column_is_high_and_has_no_separate_missing_issue():
    frame = pd.DataFrame({"x": normal_values(50), "ghost": [np.nan] * 50})

    issue = issue_for(frame, IssueKind.EMPTY_COLUMN, "ghost")

    assert issue.severity is Severity.HIGH
    assert issue.fix.safe_auto is True
    assert issues_of(frame, IssueKind.MISSING_VALUES) == []


def test_clean_data_has_no_missing_issue():
    assert issues_of(pd.DataFrame({"x": normal_values(100)}), IssueKind.MISSING_VALUES) == []


# ---------------------------------------------------------------- disguised missing


def test_placeholder_text_is_reported_with_counts_and_safe_fix():
    cities = ["Delhi", "Pune", "Mumbai", "Chennai"] * 20 + ["?"] * 5 + ["-"] * 3 + ["  "] * 2
    frame = pd.DataFrame({"city": cities})

    issue = issue_for(frame, IssueKind.DISGUISED_MISSING, "city")

    assert issue.details["count"] == 10
    assert issue.details["tokens"] == {"?": 5, "-": 3, "<blank>": 2}
    assert issue.fix.action is FixAction.REPLACE_WITH_NAN
    assert issue.fix.safe_auto is True
    assert set(issue.fix.params["tokens"]) == {"?", "-", ""}


def test_many_placeholders_are_high_severity():
    cities = ["Delhi", "Pune"] * 20 + ["unknown"] * 20
    frame = pd.DataFrame({"city": cities})

    assert issue_for(frame, IssueKind.DISGUISED_MISSING, "city").severity is Severity.HIGH


def test_placeholder_matching_ignores_case_and_padding():
    cities = ["Delhi", "Pune"] * 30 + [" N/A ", "Unknown", "NULL"]
    frame = pd.DataFrame({"city": cities})

    assert issue_for(frame, IssueKind.DISGUISED_MISSING, "city").details["count"] == 3


def test_numeric_columns_never_get_a_placeholder_issue():
    assert issues_of(pd.DataFrame({"x": normal_values(100)}), IssueKind.DISGUISED_MISSING) == []


def test_columns_that_miss_together_are_noted():
    rng = np.random.default_rng(1)
    gone = rng.random(300) < 0.2
    first = normal_values(300, seed=2)
    second = normal_values(300, seed=3)
    first[gone] = np.nan
    second[gone] = np.nan

    found = issues_of(pd.DataFrame({"a": first, "b": second}), IssueKind.MISSING_TOGETHER)

    assert len(found) == 1
    assert found[0].severity is Severity.INFO


def test_independent_gaps_are_not_noted():
    rng = np.random.default_rng(1)
    first = normal_values(300, seed=2)
    second = normal_values(300, seed=3)
    first[rng.random(300) < 0.2] = np.nan
    second[rng.random(300) < 0.2] = np.nan

    assert issues_of(pd.DataFrame({"a": first, "b": second}), IssueKind.MISSING_TOGETHER) == []


# ---------------------------------------------------------------- outliers


def test_extreme_values_are_reported_with_fences_and_examples():
    values = normal_values(500)
    values[:3] = [400.0, -300.0, 350.0]

    issue = issue_for(pd.DataFrame({"x": values}), IssueKind.OUTLIERS, "x")

    assert issue.severity is Severity.MEDIUM
    assert issue.details["extreme_count"] == 3
    assert 400.0 in issue.details["examples"]
    assert issue.fix.action is FixAction.CLIP_OUTLIERS
    assert issue.fix.params["upper"] == pytest.approx(issue.details["extreme_upper_fence"])
    assert issue.fix.safe_auto is False


def test_clean_normal_data_has_no_outlier_issue():
    assert issues_of(pd.DataFrame({"x": normal_values(1000)}), IssueKind.OUTLIERS) == []


def test_many_extreme_values_are_high_severity():
    values = normal_values(200)
    values[:10] = 1000.0  # 5% of rows

    assert issue_for(pd.DataFrame({"x": values}), IssueKind.OUTLIERS, "x").severity is Severity.HIGH


def test_only_mild_outliers_are_low_severity():
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.uniform(40, 60, 100), [78.0] * 5])

    issue = issue_for(pd.DataFrame({"x": values}), IssueKind.OUTLIERS, "x")

    assert issue.severity is Severity.LOW
    assert issue.details["extreme_count"] == 0


def test_column_with_mostly_identical_values_is_not_flagged():
    values = np.array([5.0] * 95 + [6.0, 7.0, 8.0, 9.0, 10.0])

    assert issues_of(pd.DataFrame({"x": values}), IssueKind.OUTLIERS) == []


def test_one_impossible_value_is_not_blamed_on_skew():
    ages = np.random.default_rng(6).integers(18, 70, 300).astype(float)
    ages[10] = 250.0

    issue = issue_for(pd.DataFrame({"age": ages}), IssueKind.OUTLIERS, "age")

    assert "skewed" not in issue.message
    assert issues_of(pd.DataFrame({"age": ages}), IssueKind.SKEWED) == []


def test_outliers_in_numbers_stored_as_text_are_found():
    values = [f"{value:,.2f}" for value in normal_values(300)]
    values[0] = "9,999,999.00"

    issue = issue_for(pd.DataFrame({"amount": values}), IssueKind.OUTLIERS, "amount")

    assert issue.details["extreme_count"] >= 1


# ---------------------------------------------------------------- sentinels


def test_minus_999_is_reported_as_a_sentinel_and_not_as_an_outlier():
    values = np.concatenate([np.random.default_rng(7).integers(20, 60, 200), [-999] * 10])
    frame = pd.DataFrame({"age": values.astype(float)})

    issue = issue_for(frame, IssueKind.SENTINEL_VALUES, "age")

    assert issue.details["values"] == [-999.0]
    assert issue.details["count"] == 10
    assert issue.fix.action is FixAction.REPLACE_WITH_NAN
    assert issue.fix.params["values"] == [-999.0]
    assert issues_of(frame, IssueKind.OUTLIERS) == []


def test_999_inside_the_normal_range_is_not_a_sentinel():
    prices = np.random.default_rng(8).integers(500, 1500, 300).astype(float)
    prices[:20] = 999.0

    assert issues_of(pd.DataFrame({"price": prices}), IssueKind.SENTINEL_VALUES) == []


def test_a_rare_sentinel_below_the_minimum_count_is_ignored():
    values = np.concatenate([np.random.default_rng(9).integers(20, 60, 200), [-999, -999]])

    assert issues_of(pd.DataFrame({"age": values.astype(float)}), IssueKind.SENTINEL_VALUES) == []


# ---------------------------------------------------------------- skew


def test_lognormal_data_is_reported_as_skewed_with_log_fix():
    values = np.random.default_rng(10).lognormal(3, 1, 500)

    issue = issue_for(pd.DataFrame({"income": values}), IssueKind.SKEWED, "income")

    assert issue.severity in (Severity.LOW, Severity.INFO)
    assert issue.fix.action is FixAction.LOG_TRANSFORM


def test_symmetric_data_is_not_skewed():
    assert issues_of(pd.DataFrame({"x": normal_values(500)}), IssueKind.SKEWED) == []


# ---------------------------------------------------------------- correlation


def test_near_duplicate_columns_are_reported():
    first = normal_values(300)
    second = first * 2 + np.random.default_rng(11).normal(0, 0.5, 300)
    other = normal_values(300, seed=99)

    found = issues_of(
        pd.DataFrame({"a": first, "b": second, "c": other}), IssueKind.HIGH_CORRELATION
    )

    assert len(found) == 1
    assert found[0].columns == ["a", "b"]
    assert found[0].severity is Severity.MEDIUM
    assert found[0].fix.column == "b"


def test_independent_columns_are_not_reported():
    frame = pd.DataFrame({"a": normal_values(300, seed=1), "b": normal_values(300, seed=2)})

    assert issues_of(frame, IssueKind.HIGH_CORRELATION) == []


# ---------------------------------------------------------------- numbers and dates as text


def test_numbers_as_text_are_reported_with_unreadable_examples():
    amounts = [f"₹{value:,.2f}" for value in np.linspace(100, 5000, 98)] + ["abc", "xyz"]

    issue = issue_for(pd.DataFrame({"amount": amounts}), IssueKind.NUMERIC_AS_TEXT, "amount")

    assert issue.fix.action is FixAction.CONVERT_NUMERIC
    assert issue.fix.safe_auto is True
    assert issue.details["unreadable_count"] == 2
    assert issue.details["unreadable_examples"] == ["abc", "xyz"]


def test_day_first_dates_are_detected_from_a_day_above_twelve():
    dates = [f"{day:02d}/03/2024" for day in range(1, 29)] * 3

    issue = issue_for(pd.DataFrame({"d": dates}), IssueKind.DATE_AS_TEXT, "d")

    assert issue.details["dayfirst"] is True
    assert issue.details["ambiguous"] is False
    assert issue.fix.params == {"dayfirst": True}


def test_month_first_dates_are_detected_from_a_second_part_above_twelve():
    dates = [f"03/{day:02d}/2024" for day in range(1, 29)] * 3

    issue = issue_for(pd.DataFrame({"d": dates}), IssueKind.DATE_AS_TEXT, "d")

    assert issue.details["dayfirst"] is False
    assert issue.details["ambiguous"] is False


def test_ambiguous_dates_are_flagged_as_such():
    dates = ["03/04/2024", "05/06/2024", "07/08/2024", "09/10/2024"] * 20

    issue = issue_for(pd.DataFrame({"d": dates}), IssueKind.DATE_AS_TEXT, "d")

    assert issue.details["ambiguous"] is True
    assert "ambiguous" in issue.message


def test_iso_dates_are_not_ambiguous():
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=60)]

    issue = issue_for(pd.DataFrame({"d": dates}), IssueKind.DATE_AS_TEXT, "d")

    assert issue.details["ambiguous"] is False


def test_real_datetime_columns_have_no_text_date_issue():
    frame = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=60)})

    assert issues_of(frame, IssueKind.DATE_AS_TEXT) == []


def test_guess_dayfirst_directly():
    assert guess_dayfirst(pd.Series(["25/12/2024", "01/02/2024"])) == (True, False)
    assert guess_dayfirst(pd.Series(["12/25/2024", "01/02/2024"])) == (False, False)
    assert guess_dayfirst(pd.Series(["01/02/2024"])) == (False, True)
    assert guess_dayfirst(pd.Series(["01/02/2024"]), default=True) == (True, True)
    assert guess_dayfirst(pd.Series(["2024-01-02"])) == (False, False)


# ---------------------------------------------------------------- whitespace and case


def test_padded_text_is_reported():
    cities = ["Delhi", "Pune", "Delhi ", " Pune", "Delhi", "Pune"] * 10

    issue = issue_for(pd.DataFrame({"city": cities}), IssueKind.WHITESPACE, "city")

    assert issue.details["count"] == 20
    assert issue.fix.safe_auto is True


def test_blank_strings_are_not_counted_as_padding():
    cities = ["Delhi", "Pune"] * 30 + ["   "] * 3

    assert issues_of(pd.DataFrame({"city": cities}), IssueKind.WHITESPACE) == []


def test_case_variants_map_to_the_most_common_spelling():
    cities = ["Delhi"] * 40 + ["delhi"] * 5 + ["DELHI"] * 3 + ["Pune"] * 30

    issue = issue_for(pd.DataFrame({"city": cities}), IssueKind.CASE_VARIANTS, "city")

    assert issue.severity is Severity.MEDIUM
    assert issue.fix.params["mapping"] == {"delhi": "Delhi", "DELHI": "Delhi"}
    assert issue.details["count"] == 8
    assert issue.fix.safe_auto is True


def test_case_variant_ties_are_broken_alphabetically():
    cities = ["delhi"] * 10 + ["Delhi"] * 10 + ["Pune"] * 20

    issue = issue_for(pd.DataFrame({"city": cities}), IssueKind.CASE_VARIANTS, "city")

    assert issue.fix.params["mapping"] == {"delhi": "Delhi"}


def test_distinct_categories_are_not_merged():
    cities = ["Delhi", "Pune", "Mumbai"] * 30

    assert issues_of(pd.DataFrame({"city": cities}), IssueKind.CASE_VARIANTS) == []


# ---------------------------------------------------------------- categories


def test_rare_category_is_reported():
    flag = ["no"] * 97 + ["yes"] * 3

    issue = issue_for(pd.DataFrame({"fraud": flag}), IssueKind.CLASS_IMBALANCE, "fraud")

    assert issue.severity is Severity.LOW
    assert issue.details["minority"] == "yes"


def test_extremely_rare_category_is_medium():
    flag = ["no"] * 995 + ["yes"] * 5

    issue = issue_for(pd.DataFrame({"fraud": flag}), IssueKind.CLASS_IMBALANCE, "fraud")

    assert issue.severity is Severity.MEDIUM


def test_balanced_categories_are_not_reported():
    assert issues_of(pd.DataFrame({"g": ["a", "b"] * 50}), IssueKind.CLASS_IMBALANCE) == []


def test_many_categories_are_noted():
    rng = np.random.default_rng(12)
    labels = rng.choice([f"shop_{number}" for number in range(80)], 2000)

    issue = issue_for(pd.DataFrame({"shop": labels}), IssueKind.HIGH_CARDINALITY, "shop")

    assert issue.severity is Severity.INFO


def test_constant_and_id_columns_are_reported():
    frame = pd.DataFrame({"order_id": range(1000, 1100), "country": ["India"] * 100})

    assert issue_for(frame, IssueKind.CONSTANT_COLUMN, "country").severity is Severity.LOW
    assert issue_for(frame, IssueKind.ID_COLUMN, "order_id").severity is Severity.INFO


# ---------------------------------------------------------------- duplicates


def test_duplicate_rows_are_counted_with_examples_and_safe_fix():
    frame = pd.DataFrame({"a": range(100), "b": list("xy") * 50})
    frame = pd.concat([frame, frame.iloc[[3, 7]]], ignore_index=True)

    issue = issue_for(frame, IssueKind.DUPLICATE_ROWS)

    assert issue.details["count"] == 2
    assert issue.details["example_index"] == [100, 101]
    assert issue.severity is Severity.MEDIUM
    assert issue.fix.action is FixAction.DROP_DUPLICATE_ROWS
    assert issue.fix.safe_auto is True


def test_many_duplicates_are_high_severity():
    frame = pd.DataFrame({"a": list(range(50)) * 2})

    assert issue_for(frame, IssueKind.DUPLICATE_ROWS).severity is Severity.HIGH


def test_unique_rows_have_no_duplicate_issue():
    assert issues_of(pd.DataFrame({"a": range(50)}), IssueKind.DUPLICATE_ROWS) == []


def test_list_cells_do_not_crash_the_duplicate_check():
    frame = pd.DataFrame({"a": [[1], [2], [1]] * 10})

    assert issue_for(frame, IssueKind.DUPLICATE_ROWS).details["count"] == 28


# ---------------------------------------------------------------- multivariate outliers


def correlated_frame(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    covariance = np.full((3, 3), 0.8)
    np.fill_diagonal(covariance, 1.0)
    data = rng.multivariate_normal(np.zeros(3), covariance, size=n)
    return pd.DataFrame(data, columns=["a", "b", "c"])


def test_clean_correlated_data_has_no_unusual_rows():
    assert issues_of(correlated_frame(), IssueKind.MULTIVARIATE_OUTLIERS) == []


def test_rows_that_break_the_pattern_are_found_even_when_each_value_is_normal():
    frame = correlated_frame()
    rng = np.random.default_rng(1)
    broken = list(range(8))
    # Column a high while b and c are low, against the positive correlation.
    frame.loc[broken, "a"] = 2.6 + rng.normal(0, 0.15, 8)
    frame.loc[broken, "b"] = -2.6 + rng.normal(0, 0.15, 8)
    frame.loc[broken, "c"] = -2.6 + rng.normal(0, 0.15, 8)

    issue = issue_for(frame, IssueKind.MULTIVARIATE_OUTLIERS)

    found = set(issue.details["flagged_index"])
    assert len(found & set(broken)) >= 4
    assert issue.severity is Severity.LOW


def test_too_few_rows_skip_the_multivariate_check():
    assert issues_of(correlated_frame(n=50), IssueKind.MULTIVARIATE_OUTLIERS) == []
