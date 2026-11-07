import numpy as np
import pandas as pd
import pytest

from autoeda import Severity, compare
from autoeda.drift import population_stability_index


def frame(seed=0, n=2000, mean=50.0, cities=None, missing=0.0):
    rng = np.random.default_rng(seed)
    cities = cities or {"Delhi": 0.5, "Pune": 0.3, "Mumbai": 0.2}
    values = rng.normal(mean, 10, n)
    values[rng.random(n) < missing] = np.nan
    return pd.DataFrame(
        {
            "score": values,
            "city": rng.choice(list(cities), n, p=list(cities.values())),
        }
    )


def column(report, name):
    return next(item for item in report.columns if item.column == name)


def test_same_distribution_is_stable():
    report = compare(frame(0), frame(1))

    assert all(item.severity is Severity.INFO for item in report.columns)
    assert report.drifted() == []
    assert "No column shifted" in report.summary_text()


def test_shifted_mean_is_a_large_drift():
    report = compare(frame(0), frame(1, mean=65))

    item = column(report, "score")
    assert item.severity is Severity.HIGH
    assert item.psi > 0.25
    assert item.mean_shift_in_std == pytest.approx(1.5, abs=0.15)
    assert item.ks_statistic > 0.4
    assert column(report, "city").severity is Severity.INFO


def test_small_shift_is_moderate():
    report = compare(frame(0), frame(1, mean=54))

    assert column(report, "score").severity is Severity.MEDIUM


def test_changed_category_shares_and_new_values_are_reported():
    shares = {"Delhi": 0.2, "Pune": 0.3, "Mumbai": 0.2, "Chennai": 0.3}

    report = compare(frame(0), frame(1, cities=shares))

    item = column(report, "city")
    assert item.severity is Severity.HIGH
    assert item.new_categories == ["Chennai"]
    assert "Chennai" in item.message


def test_disappeared_category_is_reported():
    report = compare(frame(0), frame(1, cities={"Delhi": 0.6, "Pune": 0.4}))

    assert column(report, "city").gone_categories == ["Mumbai"]


def test_rising_missing_rate_is_flagged_even_if_values_look_alike():
    report = compare(frame(0), frame(1, missing=0.3))

    item = column(report, "score")
    assert item.severity is Severity.HIGH
    assert item.current_missing == pytest.approx(0.3, abs=0.03)
    assert "missing 0% -> 3" in item.message


def test_schema_changes_are_reported():
    reference = frame(0)
    current = frame(1).drop(columns=["city"])
    current["extra"] = 1.0

    report = compare(reference, current)

    assert report.removed_columns == ["city"]
    assert report.added_columns == ["extra"]
    assert report.worst is Severity.HIGH
    assert "city" in report.summary_text()


def test_a_numeric_column_that_turns_into_text_is_a_type_change():
    reference = frame(0)
    current = frame(1)
    current["score"] = current["score"].map(lambda value: f"{value:.1f} units")

    report = compare(reference, current)

    assert [(name, before) for name, before, _ in report.type_changes] == [("score", "numeric")]
    assert all(item.column != "score" for item in report.columns)


def test_too_few_rows_are_not_compared():
    report = compare(frame(0, n=10), frame(1, n=10))

    assert "not compared" in column(report, "score").message


def test_files_can_be_compared(tmp_path):
    frame(0).to_csv(tmp_path / "a.csv", index=False)
    frame(1, mean=70).to_csv(tmp_path / "b.csv", index=False)

    report = compare(tmp_path / "a.csv", tmp_path / "b.csv")

    assert column(report, "score").severity is Severity.HIGH


def test_columns_are_sorted_worst_first():
    report = compare(frame(0), frame(1, mean=65))

    assert report.columns[0].column == "score"


def test_report_round_trips_through_json():
    report = compare(frame(0), frame(1, mean=65))

    restored = type(report).model_validate_json(report.model_dump_json())

    assert restored == report


def test_psi_of_identical_shares_is_zero_and_grows_with_difference():
    even = np.array([0.5, 0.5])

    assert population_stability_index(even, even) == 0.0
    assert population_stability_index(even, np.array([0.9, 0.1])) > 0.25
