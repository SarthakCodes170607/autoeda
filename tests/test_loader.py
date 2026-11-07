from pathlib import Path

import pandas as pd
import pytest

from autoeda.loader import load_csv


def write_text(tmp_path: Path, text: str, name: str = "data.csv", encoding: str = "utf-8") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


def test_loads_a_plain_comma_file(tmp_path):
    path = write_text(tmp_path, "a,b\n1,x\n2,y\n")

    loaded = load_csv(path)

    assert list(loaded.df.columns) == ["a", "b"]
    assert len(loaded.df) == 2
    assert loaded.delimiter == ","
    assert loaded.truncated is False


def test_accepts_a_string_path(tmp_path):
    path = write_text(tmp_path, "a\n1\n")

    loaded = load_csv(str(path))

    assert len(loaded.df) == 1


def test_detects_semicolon_even_with_decimal_commas(tmp_path):
    # Typical European export: ";" separates columns and "," is the decimal mark.
    path = write_text(tmp_path, "name;price;qty\nA;1,5;2\nB;2,5;3\nC;3,5;4\n")

    loaded = load_csv(path)

    assert loaded.delimiter == ";"
    assert list(loaded.df.columns) == ["name", "price", "qty"]
    assert loaded.df.loc[0, "price"] == "1,5"


def test_detects_tab_delimiter(tmp_path):
    path = write_text(tmp_path, "a\tb\tc\n1\t2\t3\n4\t5\t6\n")

    loaded = load_csv(path)

    assert loaded.delimiter == "\t"
    assert list(loaded.df.columns) == ["a", "b", "c"]


def test_detects_pipe_delimiter(tmp_path):
    path = write_text(tmp_path, "a|b|c\n1|2|3\n4|5|6\n")

    loaded = load_csv(path)

    assert loaded.delimiter == "|"
    assert loaded.df.shape == (2, 3)


def test_explicit_delimiter_overrides_detection(tmp_path):
    path = write_text(tmp_path, "a;b\n1;2\n")

    loaded = load_csv(path, delimiter=";")

    assert list(loaded.df.columns) == ["a", "b"]


def test_single_column_file_falls_back_to_comma(tmp_path):
    path = write_text(tmp_path, "only\n1\n2\n3\n")

    loaded = load_csv(path)

    assert loaded.delimiter == ","
    assert list(loaded.df.columns) == ["only"]


def test_quoted_commas_stay_inside_the_field(tmp_path):
    path = write_text(tmp_path, 'name,city\n"Smith, John",Delhi\n"Rao, Anil",Pune\n')

    loaded = load_csv(path)

    assert loaded.df.loc[0, "name"] == "Smith, John"
    assert loaded.df.shape == (2, 2)


def test_byte_order_mark_is_removed_from_first_column_name(tmp_path):
    path = write_text(tmp_path, "id,value\n1,2\n", encoding="utf-8-sig")

    loaded = load_csv(path)

    assert list(loaded.df.columns) == ["id", "value"]


def test_falls_back_to_latin1_when_not_utf8(tmp_path):
    # 0xE9 is "é" in Windows-1252/latin-1 but is invalid UTF-8 on its own.
    path = write_text(tmp_path, "name\nCafé\n", encoding="cp1252")

    loaded = load_csv(path)

    assert loaded.encoding == "latin-1"
    assert loaded.df.loc[0, "name"] == "Café"


def test_utf8_file_reports_utf8_encoding(tmp_path):
    path = write_text(tmp_path, "name\nCafé\n", encoding="utf-8")

    loaded = load_csv(path)

    assert loaded.encoding == "utf-8-sig"
    assert loaded.df.loc[0, "name"] == "Café"


def test_max_rows_cuts_the_file_and_says_so(tmp_path):
    rows = "\n".join(str(number) for number in range(100))
    path = write_text(tmp_path, "n\n" + rows + "\n")

    loaded = load_csv(path, max_rows=10)

    assert len(loaded.df) == 10
    assert loaded.truncated is True


def test_max_rows_larger_than_file_is_not_truncated(tmp_path):
    path = write_text(tmp_path, "n\n1\n2\n3\n")

    loaded = load_csv(path, max_rows=3)

    assert len(loaded.df) == 3
    assert loaded.truncated is False


@pytest.mark.parametrize("bad_value", [0, -5])
def test_max_rows_must_be_positive(tmp_path, bad_value):
    path = write_text(tmp_path, "n\n1\n")

    with pytest.raises(ValueError, match="max_rows"):
        load_csv(path, max_rows=bad_value)


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.csv"):
        load_csv(tmp_path / "nope.csv")


def test_empty_file_raises_value_error(tmp_path):
    path = write_text(tmp_path, "")

    with pytest.raises(ValueError, match="empty"):
        load_csv(path)


def test_header_only_file_gives_zero_rows_and_keeps_columns(tmp_path):
    path = write_text(tmp_path, "a,b,c\n")

    loaded = load_csv(path)

    assert loaded.df.shape == (0, 3)
    assert list(loaded.df.columns) == ["a", "b", "c"]


def test_ragged_rows_give_a_clear_error(tmp_path):
    path = write_text(tmp_path, "a,b\n1,2\n3,4,5,6\n")

    with pytest.raises(ValueError, match="parse"):
        load_csv(path)


def test_loader_does_not_clean_values(tmp_path):
    # The loader must be faithful: odd placeholders and padding survive loading,
    # so that later modules can report them as problems.
    path = write_text(tmp_path, "x\n?\n  padded  \n-\nok\n")

    loaded = load_csv(path)

    values = loaded.df["x"].tolist()
    assert values == ["?", "  padded  ", "-", "ok"]


def test_pandas_standard_missing_markers_become_nan(tmp_path):
    path = write_text(tmp_path, "x,y\n1,NA\n2,\n3,5\n")

    loaded = load_csv(path)

    assert loaded.df["y"].isna().sum() == 2
    assert isinstance(loaded.df, pd.DataFrame)
