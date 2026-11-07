# Design decisions

Why things are the way they are, including what went wrong along the way.

## Principles

1. **Library first, custom last.** Statistics come from pandas, numpy, scipy and scikit-learn.
   Custom code is for judgment (what is a problem, how severe, what fix) and glue.
2. **The loader is faithful.** It never cleans or converts. If it silently turned `?` into a
   missing value, the tool could no longer report `?` as a problem.
3. **Every decision explains itself.** Each column type has a `reason`; each issue has a
   message with counts and examples; each fix has a plain description and pandas code.
4. **Thresholds live in one place.** `TypeRules` and `CheckRules` are frozen dataclasses.
5. **Nothing the model writes is executed.** The agent fills in a validated query. The
   library never calls `exec`; only the tests do, to prove generated snippets equal `apply_fix`.

## Loading and column types

| Decision | Why |
|---|---|
| `csv.Sniffer` for the delimiter | Standard library. Limited to `, ; tab \|`. |
| Try `utf-8-sig`, then `latin-1` | `utf-8-sig` strips Excel's byte-order mark; `latin-1` decodes every byte, so it never fails. |
| pandas' own missing-value tokens are kept (empty cell, `N/A`, `NA`, `null`, `NaN`) | They become real missing values when a CSV is read, so they are reported as ordinary missing values. Only markers pandas does not convert (`?`, `unknown`, `-`, `none`...) are reported as placeholder text. Passing a DataFrame that already holds the text `N/A` reports it as placeholder text. |
| `low_memory=False` | Otherwise pandas guesses types chunk by chunk and can disagree with itself. |
| Read one extra row when `max_rows` is set | Cheapest way to know the file was cut short. |
| Text tested for numbers with `pd.to_numeric` | Reuses pandas instead of a hand-written parser. |
| Text tested for dates with `pd.to_datetime(format="mixed")` after a regex shape check | Parsing alone is too forgiving; ordinary words must not become dates. |
| Codes with leading zeros are never numbers | `01234` as a number loses its zero. |
| Marker tokens (`?`, `unknown`, `N/A`) are ignored when classifying text | Otherwise one `N/A` stops a price column being recognised as numbers. |
| An ID-like name lowers the uniqueness needed to call a column an ID (0.90 instead of 0.99) | `order_id` with a few duplicated rows is still an ID. |

## Checks

* Severity is derived from how much of the data is affected (e.g. missing: 10% medium, 40% high).
* **Outliers** use Tukey fences (1.5 and 3 x IQR) on values with sentinels removed. Skewness
  in the message is computed without the extreme values, so an outlier is not blamed on skew
  that it caused itself. Mild-only outliers need at least 3% of rows and 5 values, because
  normal data produces about 0.7% beyond 1.5 x IQR and small tables flagged themselves.
* **Sentinels** (-999, 9999, ...) must be on a short list, occur at least 3 times and
  0.5% of rows, and lie outside the range of the other values. `-1`, `0` and `99` are
  deliberately not on the list: they are too often real values.
* **Dates as text** guess day-first versus month-first from the data (any day above 12
  settles it); if nothing does, the issue says the format is ambiguous.
* **Case variants** map each spelling to the most common one (ties broken alphabetically).
* **Missing together** looks at correlation between missing-value masks, to spot a column
  group that fails as a unit (e.g. one export step).

### Unusual rows: what was tried and what went wrong

The first version used Isolation Forest with a robust z-score on its scores. On clean
Gaussian data with a threshold of 8 it was quiet, but on moderately correlated data
(r around 0.7) its recall was 0 at that threshold, and at 6 it flagged 2% of clean rows.
The scores of an Isolation Forest have no reference distribution, so any threshold was a guess.

The replacement maps each column to a normal shape by rank, then uses the squared
Mahalanobis distance, which follows a chi-square distribution under a normal model, so the
cut-off is a chosen false-alarm rate (1e-4). Two further things went wrong on the way:
scikit-learn's `QuantileTransformer` sends the single largest value in every column to
about 5.2 standard deviations, which made ordinary extremes look unusual (hence a small
rank-based transform written with `np.interp`); and a minimum-covariance-determinant
estimate gave inflated distances on finite samples, so the plain estimate (with a
pseudo-inverse) is used, which is safe because ranks bound every column.

Known weakness, shown in the benchmark: it only finds rows clearly off the data's overall
shape. With 30 correlated columns (breast cancer) it finds none of the injected rows; if
more than 2% of rows look unusual the data is not one compact cloud and nothing is reported.
Near-duplicate columns (|r| >= 0.95) are removed first.

## Fixes

* **Order matters.** Replace placeholder text, strip spaces, convert to numbers, convert to
  dates, normalise case, replace sentinel numbers, drop duplicates, drop columns, then
  impute, clip, log-transform. Rows that differ only by a space become duplicates once
  stripped, so stripping must come before the duplicate drop; imputing before dropping
  columns would waste work.
* **Safe versus judgment** is a flag on each fix (`safe_auto`). Judgment fixes are never
  applied unless asked for.
* `apply_fix` and `fix_to_code` are two implementations of one behaviour; the tests run both on
  the same data and compare. Fixes that cannot run (a column already dropped, imputing a
  text column) raise `FixError` and are reported as skipped instead of aborting.
* Clipping bounds are shown clamped to the observed range, so a bound that changes nothing
  is not presented as if it did.

## Drift

Population Stability Index on decile bins of the reference data (standard thresholds 0.10
and 0.25), plus the KS statistic, mean shift in reference standard deviations, new and
vanished categories and the change in missing share. Severity comes from PSI and missing
share, not from the KS p-value, which is tiny for any shift on large samples.

## Agent

* A LangGraph state graph; edges decide on errors, not the model. A failed query goes back to
  the planner with the error text, at most `max_attempts` times. The critic can send the plan back once.
* Planner and critic ask for JSON constrained by the pydantic schema (Ollama structured
  output); one retry shows the model its own validation error.
* **Number guard.** Numbers in the written answer are compared with the result table
  (rounding, percentage and thousands forms accepted); numbers from the question and small
  counting words are allowed. Failure returns a plain answer built without the model.
* `FakeLLM` makes the whole pipeline testable offline and allows the oracle benchmark.
* Missing values never satisfy a filter (`!=` included), as in SQL.

## Benchmarks: honesty notes

* Injected problems favour the detectors because I know what each detector looks for. The
  tables report how the checks behave when their assumptions hold, plus where they do not
  (hard outliers 59%, multivariate recall). Trials where the base table already had the
  problem in the chosen column are skipped rather than counted.
* Two benchmark bugs were found by reading failures, not by tuning detectors: a case-variant
  injection that changed nothing for single-letter labels, and sentinel injections of 9999
  into a column that legitimately contained 9999.
* The agent benchmark with a scripted planner measures the value of cleaning, not of any model.
* Speed numbers are from a 2-core cloud container.

## Known limitations

* pandas turns an all-digit column such as `00123` into `123` while reading; the zeros are lost
  before the tool sees them.
* European decimals (`1,5`) in comma-separated files are not understood.
* Type rules are heuristics and can be wrong on unusual data; every decision shows its reason.
* The Streamlit demo caps files at 50,000 rows for the free hosting tier.
