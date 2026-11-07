"""Detector benchmark: inject known problems into clean tables and measure what autoeda finds.

How it works
------------
1. Take a base table (four scikit-learn datasets plus two synthetic ones). A text column
   and a date column are added to each so every kind of problem has somewhere to go.
2. Profile the untouched table. Whatever it reports is "pre-existing" and is never counted
   for or against the detectors.
3. Inject one known problem into a randomly chosen column, at an easy and at a hard level.
4. Profile again and check whether the matching issue appears for that column.

Recall = injected problems found / injected problems.
Precision = reported issues of that kind that match an injection / all issues of that kind
reported on the injected table that were not already there before injecting.

Multivariate outliers are scored per row instead (injected rows found, and rows wrongly
flagged on clean data), because the issue is about rows, not columns.

Run:  python benchmark/detector_bench.py [--trials 20] [--out benchmark/results]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets

from autoeda import IssueKind, profile

LABELS = ["north", "south", "east", "west", "central"]


# ---------------------------------------------------------------- base tables


def _augment(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy().reset_index(drop=True)
    df["group"] = rng.choice(LABELS, len(df))
    df["event_date"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 700, len(df)), unit="D"
    )
    return df


def _sklearn(loader: Callable, name: str) -> tuple[str, pd.DataFrame]:
    frame = loader(as_frame=True).data
    return name, _augment(frame, seed=len(name))


def _customers(n: int = 2000) -> tuple[str, pd.DataFrame]:
    rng = np.random.default_rng(10)
    age = rng.integers(18, 75, n).astype(float)
    income = np.exp(rng.normal(10.5, 0.5, n)) + age * 100
    spend = income * 0.05 + rng.normal(0, 400, n)
    frame = pd.DataFrame(
        {
            "age": age,
            "income": income.round(2),
            "spend": spend.round(2),
            "visits": rng.poisson(12, n).astype(float),
            "city": rng.choice(["Delhi", "Pune", "Mumbai", "Chennai", "Kolkata", "Jaipur"], n),
            "plan": rng.choice(["free", "plus", "pro"], n, p=[0.6, 0.3, 0.1]),
        }
    )
    return "customers", _augment(frame, seed=11)


def _sensors(n: int = 5000) -> tuple[str, pd.DataFrame]:
    rng = np.random.default_rng(20)
    covariance = np.full((6, 6), 0.6)
    np.fill_diagonal(covariance, 1.0)
    data = rng.multivariate_normal(np.zeros(6), covariance, n) * [3, 5, 2, 8, 4, 6] + 50
    frame = pd.DataFrame(data.round(3), columns=[f"s{number}" for number in range(6)])
    frame["site"] = rng.choice(["A", "B", "C", "D"], n)
    return "sensors", _augment(frame, seed=21)


def base_tables() -> list[tuple[str, pd.DataFrame]]:
    return [
        _sklearn(datasets.load_iris, "iris"),
        _sklearn(datasets.load_wine, "wine"),
        _sklearn(datasets.load_diabetes, "diabetes"),
        _sklearn(datasets.load_breast_cancer, "breast_cancer"),
        _customers(),
        _sensors(),
    ]


# ---------------------------------------------------------------- injections


@dataclass
class Injection:
    kind: IssueKind
    df: pd.DataFrame
    columns: list[str]  # the issue is expected on one of these


Level = str  # "easy" or "hard"


def _pick(rng: np.random.Generator, columns: list[str]) -> str:
    return str(rng.choice(columns))


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_float_dtype(df[c]) and c != "event_date"]


def _text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "str"]


def _rows(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    count = max(3, int(round(n * share)))
    return rng.choice(n, count, replace=False)


def inject_missing(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df))
    share = rng.uniform(0.20, 0.40) if level == "easy" else rng.uniform(0.02, 0.05)
    df.loc[_rows(rng, len(df), share), name] = np.nan
    return Injection(IssueKind.MISSING_VALUES, df, [name])


def inject_disguised(df, rng, level):
    df = df.copy()
    name = _pick(rng, _text_columns(df))
    share = rng.uniform(0.10, 0.30) if level == "easy" else rng.uniform(0.01, 0.03)
    tokens = rng.choice(["N/A", "?", "unknown", "-", "null", "none"], size=2, replace=False)
    rows = _rows(rng, len(df), share)
    df.loc[rows, name] = rng.choice(tokens, len(rows))
    return Injection(IssueKind.DISGUISED_MISSING, df, [name])


def inject_duplicates(df, rng, level):
    share = rng.uniform(0.05, 0.10) if level == "easy" else rng.uniform(0.005, 0.01)
    extra = df.iloc[_rows(rng, len(df), share)]
    return Injection(IssueKind.DUPLICATE_ROWS, pd.concat([df, extra], ignore_index=True), [])


def inject_outliers(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df))
    mean, std = df[name].mean(), df[name].std()
    if level == "easy":
        rows, distance = _rows(rng, len(df), rng.uniform(0.01, 0.03)), rng.uniform(10, 15)
    else:
        rows, distance = _rows(rng, len(df), 0.005), rng.uniform(5, 6)
    signs = rng.choice([-1, 1], len(rows))
    df.loc[rows, name] = mean + signs * distance * std
    return Injection(IssueKind.OUTLIERS, df, [name])


def inject_sentinels(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df))
    share = rng.uniform(0.05, 0.10) if level == "easy" else rng.uniform(0.01, 0.02)
    # A sentinel inside the column's real range (9999 in an income column that already
    # contains 9999) cannot be told apart from data, so only out-of-range codes are used.
    low, high = df[name].min(), df[name].max()
    options = [c for c in (-999, -9999, 9999, 99999) if c < low or c > high]
    sentinel = float(rng.choice(options))
    df.loc[_rows(rng, len(df), share), name] = sentinel
    return Injection(IssueKind.SENTINEL_VALUES, df, [name])


def inject_numeric_as_text(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df))
    text = df[name].map(lambda value: f"${value:,.2f}")
    if level == "hard":
        rows = _rows(rng, len(df), 0.08)
        text.iloc[rows] = rng.choice(["N/A", "abc", "tbd"], len(rows))
    df[name] = text
    return Injection(IssueKind.NUMERIC_AS_TEXT, df, [name])


def inject_date_as_text(df, rng, level):
    df = df.copy()
    if level == "easy":
        df["event_date"] = df["event_date"].dt.strftime("%d/%m/%Y")
    else:
        text = df["event_date"].dt.strftime("%d %b %Y")
        rows = _rows(rng, len(df), 0.3)
        text.iloc[rows] = df["event_date"].iloc[rows].dt.strftime("%Y-%m-%d")
        df["event_date"] = text
    return Injection(IssueKind.DATE_AS_TEXT, df, ["event_date"])


def inject_whitespace(df, rng, level):
    df = df.copy()
    name = _pick(rng, _text_columns(df))
    share = rng.uniform(0.20, 0.40) if level == "easy" else rng.uniform(0.01, 0.03)
    rows = _rows(rng, len(df), share)
    df.loc[rows, name] = df.loc[rows, name].map(lambda value: f" {value} ")
    return Injection(IssueKind.WHITESPACE, df, [name])


def inject_case_variants(df, rng, level):
    df = df.copy()
    # Only columns where upper-casing changes something (not single letters like "A").
    changeable = [c for c in _text_columns(df) if df[c].astype(str).str.upper().ne(df[c]).any()]
    name = _pick(rng, changeable)
    share = rng.uniform(0.15, 0.30) if level == "easy" else rng.uniform(0.02, 0.04)
    rows = _rows(rng, len(df), share)
    df.loc[rows, name] = df.loc[rows, name].map(lambda value: str(value).upper())
    return Injection(IssueKind.CASE_VARIANTS, df, [name])


def inject_constant(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df) + _text_columns(df))
    value = df[name].dropna().iloc[0]
    df[name] = value
    if level == "hard":
        df.loc[rng.integers(0, len(df)), name] = np.nan  # one missing value, rest constant
    return Injection(IssueKind.CONSTANT_COLUMN, df, [name])


def inject_empty(df, rng, level):
    df = df.copy()
    df["ghost"] = np.nan if level == "easy" else None
    return Injection(IssueKind.EMPTY_COLUMN, df, ["ghost"])


def inject_correlated(df, rng, level):
    df = df.copy()
    name = _pick(rng, _numeric_columns(df))
    noise = 0.1 if level == "easy" else 0.25  # r about 0.995 and 0.97
    df["copy"] = df[name] * rng.uniform(1.5, 3) + rng.normal(0, df[name].std() * noise, len(df))
    return Injection(IssueKind.HIGH_CORRELATION, df, [name, "copy"])


INJECTORS: dict[IssueKind, Callable] = {
    IssueKind.MISSING_VALUES: inject_missing,
    IssueKind.DISGUISED_MISSING: inject_disguised,
    IssueKind.DUPLICATE_ROWS: inject_duplicates,
    IssueKind.OUTLIERS: inject_outliers,
    IssueKind.SENTINEL_VALUES: inject_sentinels,
    IssueKind.NUMERIC_AS_TEXT: inject_numeric_as_text,
    IssueKind.DATE_AS_TEXT: inject_date_as_text,
    IssueKind.WHITESPACE: inject_whitespace,
    IssueKind.CASE_VARIANTS: inject_case_variants,
    IssueKind.CONSTANT_COLUMN: inject_constant,
    IssueKind.EMPTY_COLUMN: inject_empty,
    IssueKind.HIGH_CORRELATION: inject_correlated,
}


# ---------------------------------------------------------------- scoring


def _ids(report, kind) -> set[str]:
    return {item.id for item in report.issues if item.kind is kind}


def _matches(issue, injection: Injection) -> bool:
    if injection.kind is IssueKind.DUPLICATE_ROWS:
        return True
    targets = set(injection.columns)
    return bool(targets & ({issue.column} | set(issue.columns)))


def run_injection_benchmark(trials: int) -> dict:
    tables = base_tables()
    baselines = {name: profile(df) for name, df in tables}
    skipped = 0
    tally: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"injected": 0, "found": 0, "reported": 0, "correct": 0}
    )

    for kind, injector in INJECTORS.items():
        for level in ("easy", "hard"):
            for name, base in tables:
                for trial in range(trials):
                    rng = np.random.default_rng([list(INJECTORS).index(kind), trial, len(name)])
                    injection = injector(base, rng, level)
                    report = profile(injection.df)
                    before = _ids(baselines[name], kind)
                    if kind is not IssueKind.DUPLICATE_ROWS and any(
                        _matches(item, injection)
                        for item in baselines[name].issues
                        if item.kind is kind
                    ):
                        # The table already had this problem in this column, so the trial
                        # could not tell us anything.
                        skipped += 1
                        continue

                    new_issues = [
                        item
                        for item in report.issues
                        if item.kind is kind and item.id not in before
                    ]
                    hits = [item for item in new_issues if _matches(item, injection)]
                    found = bool(hits)
                    cell = tally[(kind.value, level)]
                    cell["injected"] += 1
                    cell["found"] += int(found)
                    cell["reported"] += len(new_issues)
                    cell["correct"] += len(hits)

    rows = []
    for (kind, level), cell in tally.items():
        rows.append(
            {
                "kind": kind,
                "level": level,
                "trials": cell["injected"],
                "recall": cell["found"] / cell["injected"],
                "precision": (cell["correct"] / cell["reported"]) if cell["reported"] else None,
                "false_alarms": cell["reported"] - cell["correct"],
            }
        )
    return {"rows": rows, "skipped_trials": skipped, "baselines": _baseline_summary(baselines)}


def _baseline_summary(baselines: dict) -> dict:
    return {
        name: {
            "rows": report.n_rows,
            "issues_above_info": [item.id for item in report.issues if item.severity.rank > 0],
        }
        for name, report in baselines.items()
    }


# ---------------------------------------------------------------- multivariate


def run_multivariate_benchmark(trials: int) -> list[dict]:
    results = []
    for name, base in base_tables():
        numeric = _numeric_columns(base)
        corr = base[numeric].corr().abs().to_numpy().copy()
        np.fill_diagonal(corr, 0)
        # Use only pairs that are clearly related but not near-copies.
        usable = np.argwhere((corr > 0.5) & (corr < 0.95))
        if len(usable) == 0:
            continue
        clean_report = profile(base)
        clean_flags = sum(
            item.details["count"]
            for item in clean_report.issues
            if item.kind is IssueKind.MULTIVARIATE_OUTLIERS
        )
        found = injected = flagged_total = false_total = 0
        for trial in range(trials):
            rng = np.random.default_rng([trial, len(name)])
            a, b = usable[rng.integers(len(usable))]
            col_a, col_b = numeric[a], numeric[b]
            sign = np.sign(base[numeric].corr().iloc[a, b])
            df = base.copy()
            rows = rng.choice(len(df), max(5, len(df) // 100), replace=False)
            high_a = df[col_a].quantile(0.99)
            low_b, high_b = df[col_b].quantile(0.01), df[col_b].quantile(0.99)
            df.loc[rows, col_a] = high_a
            df.loc[rows, col_b] = low_b if sign > 0 else high_b
            report = profile(df)
            flagged: set[int] = set()
            for item in report.issues:
                if item.kind is IssueKind.MULTIVARIATE_OUTLIERS:
                    flagged = set(item.details["flagged_index"])
            injected += len(rows)
            found += len(flagged & set(int(r) for r in rows))
            flagged_total += len(flagged)
            false_total += len(flagged - set(int(r) for r in rows))
        results.append(
            {
                "dataset": name,
                "rows": len(base),
                "injected_rows": injected,
                "row_recall": found / injected,
                "rows_flagged_on_clean_table": clean_flags,
                "false_flags_per_1000_rows": 1000 * (false_total / trials) / len(base),
            }
        )
    return results


# ---------------------------------------------------------------- output


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def to_markdown(results: dict, multivariate: list[dict], trials: int) -> str:
    lines = [
        "# Detector benchmark",
        "",
        f"{trials} trials per kind, level and base table; "
        f"{len(results['baselines'])} base tables. "
        f"{results['skipped_trials']} trials were skipped because the table already had that "
        "problem in the chosen column. Easy and hard levels are defined in "
        "`benchmark/detector_bench.py`.",
        "",
        "| Problem | Level | Trials | Recall | Precision | False alarms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(results["rows"], key=lambda r: (r["kind"], r["level"])):
        lines.append(
            f"| {row['kind']} | {row['level']} | {row['trials']} | {_pct(row['recall'])} | "
            f"{_pct(row['precision'])} | {row['false_alarms']} |"
        )
    lines += [
        "",
        "## Unusual rows (multivariate)",
        "",
        "| Table | Rows | Injected rows | Row recall | Flagged on clean table | "
        "False flags per 1000 rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in multivariate:
        lines.append(
            f"| {row['dataset']} | {row['rows']} | {row['injected_rows']} | "
            f"{_pct(row['row_recall'])} | {row['rows_flagged_on_clean_table']} | "
            f"{row['false_flags_per_1000_rows']:.1f} |"
        )
    lines += ["", "## Issues reported on the untouched base tables (above INFO)", ""]
    for name, info in results["baselines"].items():
        listed = ", ".join(info["issues_above_info"]) or "none"
        lines.append(f"- **{name}** ({info['rows']} rows): {listed}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    started = time.perf_counter()
    results = run_injection_benchmark(args.trials)
    multivariate = run_multivariate_benchmark(args.trials)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detector_results.json").write_text(
        json.dumps({"detectors": results, "multivariate": multivariate}, indent=2),
        encoding="utf-8",
    )
    markdown = to_markdown(results, multivariate, args.trials)
    (args.out / "detector_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Finished in {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
