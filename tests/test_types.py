import numpy as np
import pandas as pd
import pytest

from autoeda.types import (
    ColumnKind,
    TypeRules,
    infer_column_type,
    infer_types,
    types_table,
)


def infer(values, name="value", rules=None, dtype=None):
    series = pd.Series(values, dtype=dtype)
    return infer_column_type(series, name, rules)


# ---------------------------------------------------------------- numeric


def test_decimal_measurements_are_numeric():
    rng = np.random.default_rng(0)

    result = infer(rng.normal(50, 10, 200))

    assert result.kind is ColumnKind.NUMERIC
    assert result.stored_as_text is False


def test_integers_with_many_distinct_values_are_numeric():
    rng = np.random.default_rng(1)
    ages = rng.integers(18, 80, 300)

    result = infer(ages, name="age")

    assert result.kind is ColumnKind.NUMERIC


def test_integers_with_few_distinct_values_are_categorical():
    rng = np.random.default_rng(2)
    classes = rng.integers(1, 4, 300)

    result = infer(classes, name="pclass")

    assert result.kind is ColumnKind.CATEGORICAL


def test_unique_whole_numbers_without_id_name_are_numeric():
    result = infer(range(1000, 1200), name="salary")

    assert result.kind is ColumnKind.NUMERIC


# ---------------------------------------------------------------- identifiers


@pytest.mark.parametrize("name", ["customer_id", "customerId", "ID", "Unnamed: 0", "order key"])
def test_unique_whole_numbers_with_id_like_name_are_id(name):
    result = infer(range(1000, 1200), name=name)

    assert result.kind is ColumnKind.ID


def test_id_column_with_a_few_duplicated_rows_is_still_an_id():
    # 300 distinct ids, then 6 rows repeated: 98% distinct, but the name says "id".
    ids = [*range(5000, 5300), 5004, 5020, 5044, 5080, 5120, 5200]

    assert infer(ids, name="order_id").kind is ColumnKind.ID


def test_text_id_column_with_a_few_duplicates_is_still_an_id():
    ids = [f"C{number:04d}" for number in range(100)]
    ids = [*ids, "C0001", "C0002", "C0003"]

    assert infer(ids, name="customer_id").kind is ColumnKind.ID


def test_similar_duplication_without_an_id_name_is_not_an_id():
    values = [*range(5000, 5300), 5004, 5020, 5044, 5080, 5120, 5200]

    assert infer(values, name="reading").kind is ColumnKind.NUMERIC


def test_id_name_with_mostly_repeated_values_is_not_an_id():
    # Only 5 distinct "ids" over 100 rows: that is a code or a category, not an identifier.
    values = [1000, 2000, 3000, 4000, 5000] * 20

    assert infer(values, name="store_id").kind is not ColumnKind.ID


def test_name_containing_id_letters_inside_a_word_is_not_an_id_name():
    # "width" and "valid" contain "id" but are not identifiers.
    assert infer(range(1000, 1200), name="width").kind is ColumnKind.NUMERIC
    assert infer(range(1000, 1200), name="valid").kind is ColumnKind.NUMERIC


def test_zip_codes_with_leading_zeros_are_not_read_as_numbers():
    codes = [f"0{number}" for number in range(1000, 1050)]

    result = infer(codes, name="zip")

    assert result.kind is not ColumnKind.NUMERIC
    assert result.kind is ColumnKind.ID


def test_unique_equal_length_codes_without_spaces_are_id():
    codes = [f"AB{number:05d}" for number in range(50)]

    result = infer(codes, name="reference")

    assert result.kind is ColumnKind.ID


# ---------------------------------------------------------------- numbers stored as text


def test_currency_text_is_numeric_stored_as_text_with_parse_ratio():
    rng = np.random.default_rng(3)
    good = [f"${amount:,.2f}" for amount in rng.uniform(10, 5000, 98)]
    values = [*good, "abc", "xyz"]

    result = infer(values, name="amount")

    assert result.kind is ColumnKind.NUMERIC
    assert result.stored_as_text is True
    assert result.parse_ratio == pytest.approx(0.98)
    assert "stored as text" in result.reason


def test_plain_digit_strings_are_numeric_stored_as_text():
    values = [str(number / 4) for number in range(80)]

    result = infer(values, name="reading")

    assert result.kind is ColumnKind.NUMERIC
    assert result.stored_as_text is True
    assert result.parse_ratio == 1.0


def test_text_with_too_few_numbers_stays_text_like():
    values = ["12", "abc", "def", "ghi", "jkl", "mno", "pqr", "stu", "vwx", "yz"] * 10

    result = infer(values, name="mixed")

    assert result.kind is not ColumnKind.NUMERIC
    assert result.stored_as_text is False


def test_missing_value_markers_do_not_stop_a_column_being_numeric():
    values = [str(number * 1.5) for number in range(85)] + ["?"] * 15

    result = infer(values, name="reading")

    assert result.kind is ColumnKind.NUMERIC
    assert result.stored_as_text is True
    assert result.parse_ratio == 1.0


def test_yes_no_with_unknown_markers_is_still_boolean():
    result = infer(["yes", "no", "unknown"] * 20, name="flag")

    assert result.kind is ColumnKind.BOOLEAN


def test_column_made_only_of_markers_is_empty():
    result = infer(["?", "-", "unknown", "N/A"] * 10, name="junk")

    assert result.kind is ColumnKind.EMPTY


# ---------------------------------------------------------------- booleans


def test_pandas_bool_dtype_is_boolean():
    result = infer([True, False, True, True, False] * 10)

    assert result.kind is ColumnKind.BOOLEAN


def test_yes_no_text_is_boolean_even_with_mixed_case_and_missing_values():
    values = ["Yes", "no", "YES", "No", None, "yes"] * 10

    result = infer(values, name="is_returned")

    assert result.kind is ColumnKind.BOOLEAN
    assert result.n_missing == 10


def test_true_false_text_is_boolean():
    result = infer(["True", "False"] * 20)

    assert result.kind is ColumnKind.BOOLEAN


def test_zero_one_integers_are_boolean():
    result = infer([0, 1, 1, 0, 1] * 20)

    assert result.kind is ColumnKind.BOOLEAN


def test_two_unrelated_words_are_categorical_not_boolean():
    result = infer(["red", "blue"] * 20)

    assert result.kind is ColumnKind.CATEGORICAL


# ---------------------------------------------------------------- dates


def test_datetime_dtype_is_datetime():
    result = infer(pd.date_range("2024-01-01", periods=60))

    assert result.kind is ColumnKind.DATETIME
    assert result.stored_as_text is False


def test_iso_date_text_is_datetime_stored_as_text():
    values = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=80)]

    result = infer(values, name="order_date")

    assert result.kind is ColumnKind.DATETIME
    assert result.stored_as_text is True


def test_day_first_slash_dates_are_datetime():
    values = [d.strftime("%d/%m/%Y") for d in pd.date_range("2024-01-01", periods=80)]

    result = infer(values, name="date")

    assert result.kind is ColumnKind.DATETIME


def test_month_name_dates_are_datetime():
    values = [d.strftime("%d %b %Y") for d in pd.date_range("2024-01-01", periods=80)]

    result = infer(values, name="date")

    assert result.kind is ColumnKind.DATETIME


def test_dates_with_a_few_bad_entries_still_count_as_dates():
    values = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=95)]
    values = [*values, "??", "xx", "yy", "zz", "ww"]

    result = infer(values, name="date")

    assert result.kind is ColumnKind.DATETIME
    assert result.parse_ratio == pytest.approx(0.95, abs=0.03)


def test_text_that_only_sometimes_contains_dates_is_not_datetime():
    values = ["called on 2024-01-05", "no answer", "left voicemail", "asked to retry later"] * 20

    result = infer(values, name="note")

    assert result.kind is not ColumnKind.DATETIME


# ---------------------------------------------------------------- constant and empty


def test_constant_text_column():
    result = infer(["India"] * 50, name="country")

    assert result.kind is ColumnKind.CONSTANT
    assert "India" in result.reason


def test_constant_number_column():
    result = infer([5] * 50)

    assert result.kind is ColumnKind.CONSTANT


def test_single_row_column_is_not_called_constant():
    result = infer([5])

    assert result.kind is not ColumnKind.CONSTANT


def test_all_missing_is_empty():
    result = infer([np.nan] * 30)

    assert result.kind is ColumnKind.EMPTY
    assert result.n_missing == 30


def test_all_whitespace_text_is_empty():
    result = infer(["", "  ", "\t"] * 10)

    assert result.kind is ColumnKind.EMPTY


# ---------------------------------------------------------------- text and categories


def test_few_repeated_cities_are_categorical():
    rng = np.random.default_rng(4)
    cities = rng.choice(["Delhi", "Mumbai", "Pune", "Chennai"], 300)

    result = infer(cities, name="city")

    assert result.kind is ColumnKind.CATEGORICAL
    assert result.n_unique == 4


def test_long_sentences_are_text():
    values = [f"customer number {i} said the delivery was late again" for i in range(100)]
    values = values + values[:30]

    result = infer(values, name="notes")

    assert result.kind is ColumnKind.TEXT


def test_unique_names_without_id_column_name_are_text():
    values = [f"user_{number}" for number in range(1, 150)]

    result = infer(values, name="username")

    assert result.kind is ColumnKind.TEXT


def test_category_dtype_is_categorical():
    series = pd.Series(["a", "b", "c", "a", "b"] * 10, dtype="category")

    result = infer_column_type(series, "grade")

    assert result.kind is ColumnKind.CATEGORICAL


# ---------------------------------------------------------------- bookkeeping


def test_counts_exclude_missing_values():
    result = infer(["a", None, "b", "a", None, "b", "a"], name="letter")

    assert result.n_rows == 7
    assert result.n_missing == 2
    assert result.n_unique == 2


def test_every_result_has_a_reason():
    frame = pd.DataFrame(
        {
            "n": np.arange(100) * 1.5,
            "c": ["x", "y"] * 50,
            "e": [np.nan] * 100,
            "k": ["same"] * 100,
        }
    )

    for item in infer_types(frame):
        assert item.reason


def test_infer_types_keeps_column_order_and_covers_every_column():
    frame = pd.DataFrame({"z": [1.5, 2.5, 3.5], "a": ["x", "y", "z"], "m": [True, False, True]})

    names = [item.name for item in infer_types(frame)]

    assert names == ["z", "a", "m"]


def test_infer_types_handles_duplicate_column_names():
    frame = pd.DataFrame([[1.5, "x"], [2.5, "y"], [3.5, "z"]], columns=["a", "a"])

    results = infer_types(frame)

    assert len(results) == 2
    assert results[0].name == results[1].name == "a"
    assert results[0].kind is ColumnKind.NUMERIC


def test_table_with_no_rows_does_not_crash():
    frame = pd.DataFrame({"a": pd.Series([], dtype="float64"), "b": pd.Series([], dtype="object")})

    results = infer_types(frame)

    assert [item.kind for item in results] == [ColumnKind.EMPTY, ColumnKind.EMPTY]


def test_mixed_python_objects_do_not_crash():
    values = [1, "two", 3.5, None, "four"] * 10

    result = infer(values, name="odd")

    assert result.kind in set(ColumnKind)


def test_custom_rules_change_the_decision():
    rng = np.random.default_rng(5)
    classes = rng.integers(1, 6, 1000)

    default_result = infer(classes, name="rating")
    strict_rules = TypeRules(categorical_max_unique=3)
    strict_result = infer(classes, name="rating", rules=strict_rules)

    assert default_result.kind is ColumnKind.CATEGORICAL
    assert strict_result.kind is ColumnKind.NUMERIC


def test_types_table_has_one_row_per_column():
    frame = pd.DataFrame({"a": [1.5, 2.5, 3.5], "b": ["x", "y", "z"]})

    table = types_table(infer_types(frame))

    assert list(table["column"]) == ["a", "b"]
    assert {"kind", "reason", "missing", "unique"} <= set(table.columns)
