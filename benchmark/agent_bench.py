"""Agent benchmark: do answers stay correct when the data is messy?

A clean sales table is generated, its correct answers are computed with plain pandas, and
then the table is made messy in ways real exports are messy (money stored as text with a
currency symbol, mixed upper/lower case, padded text, dd/mm/yyyy dates, duplicated rows,
-999 placeholders). The same questions are asked of the messy table in two ways:

  raw      the query runs on the messy table as it is (what a plain text-to-pandas agent sees)
  autoeda  the pipeline profiles and cleans first, then runs the query

An answer is correct when the result table matches the ground truth (1e-6 relative, 2% for
the one question whose placeholders destroy information).

Two planners:

  oracle   a scripted planner that always asks the right question. This isolates what the
           cleaning step is worth. It needs no model and is fully reproducible.
  ollama   a real local model plans each query. Run this on your machine:
           ollama pull qwen2.5:7b-instruct
           python benchmark/agent_bench.py --planner ollama --model qwen2.5:7b-instruct

Run:  python benchmark/agent_bench.py --planner oracle --seeds 5
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from autoeda.agent import FakeLLM, OllamaClient, ask
from autoeda.agent.queries import QueryResult

REGIONS = ["north", "south", "east", "west", "central"]
CATEGORIES = ["electronics", "grocery", "clothing", "home", "toys"]
PRODUCTS = [f"product_{number:02d}" for number in range(25)]


# ---------------------------------------------------------------- data


def make_clean(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quantity = rng.integers(1, 9, n)
    unit_price = rng.choice([4.5, 9.99, 19.0, 49.5, 120.0, 249.99], n)
    return pd.DataFrame(
        {
            "order_id": np.arange(100_000, 100_000 + n),
            "customer_id": rng.integers(1000, 1000 + n // 3, n),
            "region": rng.choice(REGIONS, n),
            "category": rng.choice(CATEGORIES, n),
            "product": rng.choice(PRODUCTS, n),
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": (quantity * unit_price).round(2),
            "order_date": pd.Timestamp("2023-01-01")
            + pd.to_timedelta(rng.integers(0, 730, n), "D"),
            "returned": rng.choice(["yes", "no"], n, p=[0.1, 0.9]),
            "age": rng.integers(18, 80, n).astype(float),
        }
    )


def make_messy(clean: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    df = clean.copy()
    n = len(df)
    df["amount"] = df["amount"].map(lambda value: f"₹{value:,.2f}")
    variants = rng.choice(["upper", "title"], n)
    flip = rng.random(n) < 0.15
    df["region"] = [
        (value.upper() if kind == "upper" else value.title()) if hit else value
        for value, kind, hit in zip(df["region"], variants, flip, strict=True)
    ]
    padded = rng.random(n) < 0.20
    df["category"] = [
        f"  {v} " if hit else v for v, hit in zip(df["category"], padded, strict=True)
    ]
    df["order_date"] = df["order_date"].dt.strftime("%d/%m/%Y")
    df.loc[rng.random(n) < 0.04, "age"] = -999.0
    extra = df.sample(frac=0.05, random_state=seed)
    return pd.concat([df, extra]).sample(frac=1, random_state=seed + 1).reset_index(drop=True)


# ---------------------------------------------------------------- questions

Truth = Callable[[pd.DataFrame], dict[tuple, float]]


@dataclass
class Question:
    id: str
    text: str
    plan: dict
    truth: Truth
    tolerance: float = 1e-6


def _grouped(df, by, column, agg):
    series = df.groupby(by)[column].agg(agg)
    return {(key,) if not isinstance(key, tuple) else key: float(v) for key, v in series.items()}


def _q(op, column=None, **extra):
    return {"operation": op, "column": column, **extra}


QUESTIONS = [
    Question(
        "total_sales",
        "What is the total sales amount?",
        {"fix_ids": [], "query": _q("aggregate", "amount", agg="sum")},
        lambda c: {(): float(c["amount"].sum())},
    ),
    Question(
        "sales_by_region",
        "What are the total sales for each region?",
        {"fix_ids": [], "query": _q("aggregate", "amount", agg="sum", group_by=["region"])},
        lambda c: _grouped(c, "region", "amount", "sum"),
    ),
    Question(
        "top_categories_avg",
        "Which three categories have the highest average order amount?",
        {"fix_ids": [], "query": _q("top_k", "amount", agg="mean", group_by=["category"], k=3)},
        lambda c: dict(
            sorted(_grouped(c, "category", "amount", "mean").items(), key=lambda kv: -kv[1])[:3]
        ),
    ),
    Question(
        "order_count",
        "How many orders are there?",
        {"fix_ids": [], "query": _q("aggregate", "order_id", agg="count")},
        lambda c: {(): float(len(c))},
    ),
    Question(
        "unique_customers",
        "How many different customers placed orders?",
        {"fix_ids": [], "query": _q("aggregate", "customer_id", agg="nunique")},
        lambda c: {(): float(c["customer_id"].nunique())},
    ),
    Question(
        "h1_2024_sales",
        "What was the total sales amount in the first half of 2024?",
        {
            "fix_ids": [],
            "query": _q(
                "aggregate",
                "amount",
                agg="sum",
                filters=[
                    {"column": "order_date", "op": ">=", "value": "2024-01-01"},
                    {"column": "order_date", "op": "<", "value": "2024-07-01"},
                ],
            ),
        },
        lambda c: {
            (): float(
                c.loc[
                    (c["order_date"] >= "2024-01-01") & (c["order_date"] < "2024-07-01"), "amount"
                ].sum()
            )
        },
    ),
    Question(
        "monthly_trend",
        "Show the monthly sales trend.",
        {
            "fix_ids": [],
            "query": _q("trend", "amount", agg="sum", date_column="order_date", freq="M"),
        },
        lambda c: {
            (str(k),): float(v)
            for k, v in c.groupby(c["order_date"].dt.to_period("M"))["amount"].sum().items()
        },
    ),
    Question(
        "north_sales",
        "What are the total sales in the north region?",
        {
            "fix_ids": [],
            "query": _q(
                "aggregate",
                "amount",
                agg="sum",
                filters=[{"column": "region", "op": "==", "value": "north"}],
            ),
        },
        lambda c: {(): float(c.loc[c["region"] == "north", "amount"].sum())},
    ),
    Question(
        "average_age",
        "What is the average customer age?",
        {"fix_ids": ["sentinels:age"], "query": _q("aggregate", "age", agg="mean")},
        lambda c: {(): float(c["age"].mean())},
        tolerance=0.02,  # the true ages of the placeholder rows are gone
    ),
    Question(
        "top_products",
        "Which 5 products have the highest total sales?",
        {"fix_ids": [], "query": _q("top_k", "amount", agg="sum", group_by=["product"], k=5)},
        lambda c: dict(
            sorted(_grouped(c, "product", "amount", "sum").items(), key=lambda kv: -kv[1])[:5]
        ),
    ),
    Question(
        "median_price",
        "What is the median unit price?",
        {"fix_ids": [], "query": _q("aggregate", "unit_price", agg="median")},
        lambda c: {(): float(c["unit_price"].median())},
    ),
    Question(
        "returned_orders",
        "How many orders were returned?",
        {
            "fix_ids": [],
            "query": _q(
                "aggregate",
                "order_id",
                agg="count",
                filters=[{"column": "returned", "op": "==", "value": "yes"}],
            ),
        },
        lambda c: {(): float((c["returned"] == "yes").sum())},
    ),
    Question(
        "avg_by_category",
        "What is the average order amount in each category?",
        {"fix_ids": [], "query": _q("aggregate", "amount", agg="mean", group_by=["category"])},
        lambda c: _grouped(c, "category", "amount", "mean"),
    ),
    Question(
        "quantity_by_category",
        "How many items were sold in each category?",
        {"fix_ids": [], "query": _q("aggregate", "quantity", agg="sum", group_by=["category"])},
        lambda c: _grouped(c, "category", "quantity", "sum"),
    ),
]


# ---------------------------------------------------------------- scoring


def result_to_dict(result: QueryResult) -> dict[tuple, float]:
    table = result.table
    value_column = table.columns[-1]
    keys = list(table.columns[:-1])
    output: dict[tuple, float] = {}
    for _, row in table.iterrows():
        key = tuple(str(row[k]).strip() if k != "period" else str(row[k]) for k in keys)
        output[key] = float(row[value_column])
    return output


def is_correct(result: QueryResult | None, truth: dict[tuple, float], tolerance: float) -> bool:
    if result is None:
        return False
    got = result_to_dict(result)
    if set(got) != set(truth):
        return False
    return all(abs(got[k] - truth[k]) <= tolerance * max(abs(truth[k]), 1.0) for k in truth)


# ---------------------------------------------------------------- planners


def oracle_llm() -> FakeLLM:
    by_text = {q.text: q.plan for q in QUESTIONS}

    def respond(system: str, user: str) -> str:
        if system.startswith("You plan"):
            text = user.split("Question: ", 1)[1].splitlines()[0].strip()
            return json.dumps(by_text[text])
        if system.startswith("You check"):
            return '{"answers_question": true, "concern": ""}'
        return "Here is the result."

    return FakeLLM(respond)


# ---------------------------------------------------------------- run


def run(planner: str, model: str, seeds: int, variants: list[str]) -> dict:
    stats = {
        name: {"correct": 0, "wrong": 0, "error": 0, "plain": 0, "retries": 0, "seconds": 0.0}
        for name in variants
    }
    per_question = {name: {q.id: 0 for q in QUESTIONS} for name in variants}
    total = 0

    for seed in range(seeds):
        clean = make_clean(seed)
        messy = make_messy(clean, seed)
        for question in QUESTIONS:
            truth = question.truth(clean)
            total += 1
            for name in variants:
                llm = oracle_llm() if planner == "oracle" else OllamaClient(model=model)
                options = {"clean_data": name != "raw"}
                started = time.perf_counter()
                outcome = ask(messy, question.text, llm, **options)
                stats[name]["seconds"] += time.perf_counter() - started
                cell = stats[name]
                if outcome.result is None:
                    cell["error"] += 1
                elif is_correct(outcome.result, truth, question.tolerance):
                    cell["correct"] += 1
                    per_question[name][question.id] += 1
                else:
                    cell["wrong"] += 1
                cell["plain"] += int(outcome.plain_answer_used)
                cell["retries"] += int(sum(s["step"] == "plan" for s in outcome.trace) > 1)

    return {
        "planner": planner,
        "model": model if planner == "ollama" else None,
        "seeds": seeds,
        "questions_per_variant": total,
        "variants": stats,
        "per_question": per_question,
    }


def to_markdown(results: dict) -> str:
    n = results["questions_per_variant"]
    planner = results["planner"]
    title = "scripted oracle planner" if planner == "oracle" else f"Ollama model {results['model']}"
    lines = [
        f"# Agent benchmark ({title})",
        "",
        f"{results['seeds']} generated sales tables x {len(QUESTIONS)} questions = {n} questions "
        "per variant. Ground truth is computed with plain pandas on the clean table before it "
        "is made messy.",
        "",
        "| Variant | Correct | Wrong answer | No answer | Plain-answer fallback "
        "| Needed repair | Seconds/question |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cell in results["variants"].items():
        lines.append(
            f"| {name} | {cell['correct'] / n:.0%} ({cell['correct']}/{n}) "
            f"| {cell['wrong'] / n:.0%} | "
            f"{cell['error'] / n:.0%} | {cell['plain'] / n:.0%} | {cell['retries'] / n:.0%} | "
            f"{cell['seconds'] / n:.2f} |"
        )
    names = list(results["variants"])
    lines += ["", "## Correct answers per question", "", "| Question | " + " | ".join(names) + " |"]
    lines.append("|---|" + "---:|" * len(names))
    for question in QUESTIONS:
        counts = " | ".join(
            f"{results['per_question'][v][question.id]}/{results['seeds']}" for v in names
        )
        lines.append(f"| {question.id} | {counts} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=["oracle", "ollama"], default="oracle")
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    results = run(args.planner, args.model, args.seeds, ["raw", "autoeda"])
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"agent_results_{args.planner}"
    (args.out / f"{stem}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    markdown = to_markdown(results)
    (args.out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
