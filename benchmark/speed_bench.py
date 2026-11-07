"""Speed benchmark: how long profile() and clean() take as the table grows.

The table has 12 columns: 5 numeric, 4 categorical, 1 text-as-numbers, 1 date, 1 id,
with missing values, duplicates and outliers mixed in. Times are wall-clock seconds on the
machine that runs this, so numbers differ between computers.

Run:  python benchmark/speed_bench.py
"""

from __future__ import annotations

import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from autoeda import clean, profile


def make_table(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(100, 15, n)
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "a": base,
            "b": base * 0.5 + rng.normal(0, 20, n),
            "c": rng.exponential(50, n),
            "d": rng.integers(0, 1000, n).astype(float),
            "e": rng.normal(0, 1, n),
            "city": rng.choice(["Delhi", "Pune", "Mumbai", "Chennai", "Kolkata"], n),
            "plan": rng.choice(["free", "plus", "pro"], n),
            "segment": rng.choice(["a", "b", "c", "d"], n),
            "code": rng.choice([f"X{number}" for number in range(200)], n),
            "amount": [f"${value:,.2f}" for value in rng.uniform(5, 5000, n)],
            "when": (
                pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), "D")
            ).strftime("%d/%m/%Y"),
        }
    )
    frame.loc[rng.random(n) < 0.08, "c"] = np.nan
    frame.loc[rng.random(n) < 0.02, "a"] = 5000.0
    return pd.concat([frame, frame.sample(frac=0.01, random_state=1)], ignore_index=True)


def main() -> None:
    rows = []
    for n in (10_000, 100_000, 500_000, 1_000_000):
        frame = make_table(n)
        started = time.perf_counter()
        report = profile(frame)
        profile_seconds = time.perf_counter() - started
        started = time.perf_counter()
        clean(frame, report)
        clean_seconds = time.perf_counter() - started
        rows.append(
            {
                "rows": len(frame),
                "profile_seconds": round(profile_seconds, 1),
                "clean_seconds": round(clean_seconds, 1),
                "issues": len(report.issues),
            }
        )
        print(rows[-1], flush=True)

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    machine = f"{platform.processor() or platform.machine()}, Python {platform.python_version()}"
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    (out / "speed_results.json").write_text(
        json.dumps({"machine": machine, "peak_memory_mb": round(peak_mb), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Speed benchmark",
        "",
        f"Machine: {machine}. Peak memory for the whole run: {peak_mb:,.0f} MB.",
        "",
        "| Rows | profile() seconds | clean() seconds (includes the re-check) | Issues found |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['rows']:,} | {row['profile_seconds']} | {row['clean_seconds']} | "
            f"{row['issues']} |"
        )
    (out / "speed_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
