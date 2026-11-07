"""Tests for applying fixes, the order they run in, and the code they generate."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoeda import Fix, FixAction, clean, profile
from autoeda.fixes import FixError, apply_fix, fix_phase, fix_to_code

SAMPLE = Path(__file__).parent.parent / "examples" / "sample_messy.csv"


def run_code(df, fix):
    """Run the generated snippet on a copy of df (tests only; the library never does this)."""
    namespace = {"df": df.copy(), "pd": pd, "np": np}
    exec(fix_to_code(fix), namespace)  # noqa: S102
    return namespace["df"]


CASES = [
    (
        Fix(action=FixAction.STRIP_WHITESPACE, column="t", description=""),
        pd.DataFrame({"t": [" a", "b ", None, "  c  "]}),
    ),
    (
        Fix(
            action=FixAction.REPLACE_WITH_NAN,
            column="t",
            params={"tokens": ["?", "N/A", "", "Unknown"]},
            description="",
        ),
        pd.DataFrame({"t": ["a", "?", " n/a ", "", "UNKNOWN", None, "b"]}),
    ),
    (
        Fix(
            action=FixAction.REPLACE_WITH_NAN,
            column="n",
            params={"values": [-999.0]},
            description="",
        ),
        pd.DataFrame({"n": [1.0, -999.0, 3.0, np.nan, -999.0]}),
    ),
    (
        Fix(action=FixAction.CONVERT_NUMERIC, column="t", description=""),
        pd.DataFrame({"t": ["$1,234.50", "₹ 99", "12%", "abc", None, " 7 "]}),
    ),
    (
        Fix(
            action=FixAction.CONVERT_DATETIME,
            column="t",
            params={"dayfirst": True},
            description="",
        ),
        pd.DataFrame({"t": ["25/12/2024", "01/02/2024", " 03/04/2024 ", "junk", None]}),
    ),
    (
        Fix(
            action=FixAction.CONVERT_DATETIME,
            column="t",
            params={"dayfirst": False},
            description="",
        ),
        pd.DataFrame({"t": ["2024-03-15", "2024-12-01", None]}),
    ),
    (
        Fix(
            action=FixAction.NORMALIZE_CASE,
            column="t",
            params={"mapping": {"delhi": "Delhi", "DELHI": "Delhi"}},
            description="",
        ),
        pd.DataFrame({"t": ["Delhi", "delhi", "DELHI", "Pune", None]}),
    ),
    (
        Fix(action=FixAction.IMPUTE_MEDIAN, column="n", description=""),
        pd.DataFrame({"n": [1.0, np.nan, 5.0, 3.0]}),
    ),
    (
        Fix(action=FixAction.IMPUTE_MODE, column="t", description=""),
        pd.DataFrame({"t": ["a", "b", "b", None, "a", "b"]}),
    ),
    (
        Fix(
            action=FixAction.CLIP_OUTLIERS,
            column="n",
            params={"lower": 0.0, "upper": 10.0},
            description="",
        ),
        pd.DataFrame({"n": [-5.0, 2.0, 50.0, np.nan]}),
    ),
    (
        Fix(action=FixAction.LOG_TRANSFORM, column="n", description=""),
        pd.DataFrame({"n": [0.0, 1.0, 10.0, 1000.0]}),
    ),
    (
        Fix(action=FixAction.DROP_COLUMN, column="a", description=""),
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
    ),
    (
        Fix(action=FixAction.DROP_DUPLICATE_ROWS, description=""),
        pd.DataFrame({"a": [1, 2, 1, 2, 3], "b": list("xyxyz")}),
    ),
]


@pytest.mark.parametrize(("fix", "frame"), CASES, ids=[case[0].action.value for case in CASES])
def test_generated_code_gives_the_same_result_as_apply_fix(fix, frame):
    expected = apply_fix(frame, fix)
    produced = run_code(frame, fix)

    pd.testing.assert_frame_equal(produced, expected)


@pytest.mark.parametrize(("fix", "frame"), CASES, ids=[case[0].action.value for case in CASES])
def test_apply_fix_never_changes_its_input(fix, frame):
    before = frame.copy()

    apply_fix(frame, fix)

    pd.testing.assert_frame_equal(frame, before)


# ---------------------------------------------------------------- what each fix does


def test_stripping_keeps_missing_values_and_non_strings():
    frame = pd.DataFrame({"t": [" a ", None, 5, "b "]}, dtype=object)
    fix = Fix(action=FixAction.STRIP_WHITESPACE, column="t", description="")

    result = apply_fix(frame, fix)

    assert result["t"].tolist()[0] == "a"
    assert result["t"].tolist()[2] == 5
    assert pd.isna(result["t"].tolist()[1])


def test_placeholder_replacement_ignores_case_and_padding():
    frame = pd.DataFrame({"t": ["ok", " N/A ", "n/a", "Unknown"]})
    fix = Fix(
        action=FixAction.REPLACE_WITH_NAN,
        column="t",
        params={"tokens": ["N/A", "unknown"]},
        description="",
    )

    assert apply_fix(frame, fix)["t"].isna().tolist() == [False, True, True, True]


def test_numeric_conversion_turns_unreadable_text_into_missing():
    frame = pd.DataFrame({"t": ["$1,000", "abc", "5"]})
    fix = Fix(action=FixAction.CONVERT_NUMERIC, column="t", description="")

    assert apply_fix(frame, fix)["t"].tolist()[0] == 1000.0
    assert apply_fix(frame, fix)["t"].isna().tolist() == [False, True, False]


def test_dayfirst_changes_how_ambiguous_dates_are_read():
    frame = pd.DataFrame({"t": ["03/04/2024"]})
    day_first = Fix(
        action=FixAction.CONVERT_DATETIME, column="t", params={"dayfirst": True}, description=""
    )
    month_first = Fix(
        action=FixAction.CONVERT_DATETIME, column="t", params={"dayfirst": False}, description=""
    )

    assert apply_fix(frame, day_first)["t"].iloc[0] == pd.Timestamp("2024-04-03")
    assert apply_fix(frame, month_first)["t"].iloc[0] == pd.Timestamp("2024-03-04")


def test_dropping_duplicates_resets_the_row_labels():
    frame = pd.DataFrame({"a": [1, 1, 2]})

    result = apply_fix(frame, Fix(action=FixAction.DROP_DUPLICATE_ROWS, description=""))

    assert result.index.tolist() == [0, 1]


@pytest.mark.parametrize(
    ("action", "frame", "message"),
    [
        (FixAction.IMPUTE_MEDIAN, pd.DataFrame({"x": ["a", None]}), "not numeric"),
        (FixAction.IMPUTE_MODE, pd.DataFrame({"x": [None, None]}, dtype=object), "mode"),
        (FixAction.LOG_TRANSFORM, pd.DataFrame({"x": [-5.0, 2.0]}), "undefined"),
        (FixAction.DROP_COLUMN, pd.DataFrame({"y": [1]}), "no longer exists"),
    ],
)
def test_impossible_fixes_raise_fix_error(action, frame, message):
    fix = Fix(action=action, column="x", params={"lower": 0, "upper": 1}, description="")

    with pytest.raises(FixError, match=message):
        apply_fix(frame, fix)


def test_fix_order_runs_repairs_before_drops_before_modelling_choices():
    actions = [
        Fix(
            action=FixAction.REPLACE_WITH_NAN, column="a", params={"tokens": ["?"]}, description=""
        ),
        Fix(action=FixAction.STRIP_WHITESPACE, column="a", description=""),
        Fix(action=FixAction.CONVERT_NUMERIC, column="a", description=""),
        Fix(action=FixAction.CONVERT_DATETIME, column="a", description=""),
        Fix(action=FixAction.NORMALIZE_CASE, column="a", params={"mapping": {}}, description=""),
        Fix(action=FixAction.REPLACE_WITH_NAN, column="a", params={"values": [-1]}, description=""),
        Fix(action=FixAction.DROP_DUPLICATE_ROWS, description=""),
        Fix(action=FixAction.DROP_COLUMN, column="a", description=""),
        Fix(action=FixAction.IMPUTE_MEDIAN, column="a", description=""),
        Fix(action=FixAction.CLIP_OUTLIERS, column="a", params={}, description=""),
        Fix(action=FixAction.LOG_TRANSFORM, column="a", description=""),
    ]

    phases = [fix_phase(fix) for fix in actions]

    assert phases == sorted(phases)
    assert len(set(phases)) == len(phases)


# ---------------------------------------------------------------- clean()


def test_safe_cleaning_fixes_the_sample_file():
    result = clean(SAMPLE)

    ids = {item.issue_id for item in result.applied}
    assert ids == {
        "disguised_missing:amount",
        "numeric_as_text:amount",
        "date_as_text:order_date",
        "duplicates:rows",
        "empty_column:internal_flag",
    }
    assert len(result.df) == 300
    assert "internal_flag" not in result.df.columns
    assert pd.api.types.is_numeric_dtype(result.df["amount"])
    assert pd.api.types.is_datetime64_any_dtype(result.df["order_date"])
    assert result.after.quality_score > result.before.quality_score
    assert result.skipped == []


def test_safe_cleaning_leaves_judgment_calls_alone():
    result = clean(SAMPLE)

    # Outliers and missing values are reported, but not touched without being asked.
    assert result.df["age"].max() == 250
    assert result.df["unit_price"].isna().sum() == 12
    assert result.after.issue("outliers:age") is not None


def test_cleaning_everything_applies_the_judgment_calls_too():
    result = clean(SAMPLE, fixes="all")

    assert result.df["age"].max() < 250
    assert result.df["unit_price"].isna().sum() == 0
    assert "country" not in result.df.columns


def test_cleaning_chosen_issue_ids_only():
    result = clean(SAMPLE, fixes=["duplicates:rows"])

    assert [item.issue_id for item in result.applied] == ["duplicates:rows"]
    assert len(result.df) == 300
    assert "internal_flag" in result.df.columns


def test_unknown_issue_id_is_rejected():
    with pytest.raises(ValueError, match="No such issue id"):
        clean(SAMPLE, fixes=["nonsense:column"])


def test_skip_removes_a_fix_from_the_plan():
    result = clean(SAMPLE, skip=["duplicates:rows"])

    assert "duplicates:rows" not in {item.issue_id for item in result.applied}
    assert len(result.df) == 306


def test_cleaning_does_not_modify_the_input_frame():
    frame = pd.read_csv(SAMPLE)
    before = frame.copy()

    clean(frame)

    pd.testing.assert_frame_equal(frame, before)


def test_a_fix_that_stops_applying_is_reported_as_skipped():
    # The column is dropped by one fix, so the other fix on it cannot run.
    frame = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0] * 20, "y": [1.0, 2.0, 3.0, 4.0] * 20})
    report = profile(frame)
    drop = Fix(action=FixAction.DROP_COLUMN, column="x", description="drop x")
    impute = Fix(action=FixAction.IMPUTE_MEDIAN, column="x", description="fill x")
    report.issues[:] = [
        report.issues[0].model_copy(update={"id": "a:x", "fix": drop}),
        report.issues[0].model_copy(update={"id": "b:x", "fix": impute}),
    ]

    result = clean(frame, report, fixes="all")

    assert [item.issue_id for item in result.applied] == ["a:x"]
    assert [item.issue_id for item in result.skipped] == ["b:x"]
    assert "no longer exists" in result.skipped[0].reason


def test_generated_script_reproduces_the_cleaned_table(tmp_path):
    result = clean(SAMPLE, fixes="all")
    script = result.code.replace(repr(result.source_name), repr(str(SAMPLE)))

    namespace: dict = {}
    exec(script, namespace)  # noqa: S102

    pd.testing.assert_frame_equal(namespace["df"], result.df)


def test_summary_text_mentions_score_change():
    text = clean(SAMPLE).summary_text()

    assert text.startswith("Quality score:")
    assert "->" in text
