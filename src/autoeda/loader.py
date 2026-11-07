"""Load a CSV file into a pandas DataFrame.

Design rule: the loader is faithful. It never cleans, converts or drops values.
Problems in the data are reported by later modules, not hidden here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# How many bytes of the file we look at to guess the delimiter.
SNIFF_BYTES = 64 * 1024

# Delimiters we are willing to guess between.
CANDIDATE_DELIMITERS = ",;\t|"
DEFAULT_DELIMITER = ","

# "utf-8-sig" reads plain UTF-8 and also removes the invisible byte-order mark that
# Excel often adds at the start of a file. "latin-1" can decode any byte at all, so
# it is the last resort and can never fail.
ENCODINGS_TO_TRY = ("utf-8-sig", "latin-1")


@dataclass(frozen=True)
class LoadedDataset:
    """A loaded table together with the details of how it was read."""

    df: pd.DataFrame
    path: Path
    encoding: str
    delimiter: str
    truncated: bool  # True when max_rows cut the file short


def detect_delimiter(path: Path) -> str:
    """Guess the delimiter from the start of the file.

    Uses csv.Sniffer from the standard library. When the guess is impossible
    (for example a file with only one column), falls back to a comma.
    """
    raw_bytes = path.read_bytes()[:SNIFF_BYTES]

    # Delimiters are plain ASCII characters, so a lossy decode is fine for sniffing.
    sample_text = raw_bytes.decode("utf-8-sig", errors="replace")

    # If we hit the byte limit, the last line is probably cut in half. Drop it,
    # because a half line would confuse the sniffer.
    if len(raw_bytes) == SNIFF_BYTES:
        sample_text = sample_text.rsplit("\n", 1)[0]

    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=CANDIDATE_DELIMITERS)
    except csv.Error:
        return DEFAULT_DELIMITER

    return dialect.delimiter


def load_csv(
    path: str | Path,
    *,
    max_rows: int | None = None,
    delimiter: str | None = None,
) -> LoadedDataset:
    """Read a CSV file.

    Args:
        path: Location of the CSV file.
        max_rows: Read at most this many data rows. Useful for huge files and for
            memory-limited demos. The result says whether the file was cut short.
        delimiter: Force a delimiter. When None, it is detected automatically.

    Raises:
        FileNotFoundError: The path does not point to a file.
        ValueError: The file is empty, cannot be parsed, or max_rows is not positive.
    """
    file_path = Path(path)

    if not file_path.is_file():
        raise FileNotFoundError(f"No such file: {file_path}")

    if file_path.stat().st_size == 0:
        raise ValueError(f"File is empty: {file_path}")

    if max_rows is not None and max_rows <= 0:
        raise ValueError(f"max_rows must be a positive number, got {max_rows}")

    chosen_delimiter = delimiter if delimiter is not None else detect_delimiter(file_path)

    # Read one extra row so we can tell whether the file really had more rows.
    rows_to_read = max_rows + 1 if max_rows is not None else None

    for encoding in ENCODINGS_TO_TRY:
        try:
            frame = pd.read_csv(
                file_path,
                sep=chosen_delimiter,
                encoding=encoding,
                nrows=rows_to_read,
                # Read the whole file before deciding each column's type, so one
                # odd value late in the file cannot cause inconsistent types.
                low_memory=False,
            )
        except UnicodeDecodeError:
            # Wrong encoding guess: try the next one.
            continue
        except pd.errors.EmptyDataError as error:
            raise ValueError(f"No columns could be read from {file_path}") from error
        except pd.errors.ParserError as error:
            raise ValueError(
                f"Could not parse {file_path} as CSV (delimiter {chosen_delimiter!r}): {error}"
            ) from error

        truncated = False
        if max_rows is not None and len(frame) > max_rows:
            frame = frame.iloc[:max_rows].copy()
            truncated = True

        return LoadedDataset(
            df=frame,
            path=file_path,
            encoding=encoding,
            delimiter=chosen_delimiter,
            truncated=truncated,
        )

    # Unreachable in practice, because latin-1 decodes every possible byte.
    raise ValueError(f"Could not decode {file_path} with any supported encoding")
