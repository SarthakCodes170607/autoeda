# autoeda

Automatic exploratory data analysis for CSV files: it finds data-quality problems,
ranks them by severity, and suggests the pandas code to fix each one.

> **Status: early development.** Only the first module exists so far.

## What works today

- `load_csv`: reads a CSV with automatic delimiter and encoding detection, and never
  modifies the values it reads.
- `infer_types`: decides what each column *means* (numeric, categorical, boolean,
  datetime, text, id, constant, empty), including numbers and dates stored as text,
  and explains each decision.

```python
from autoeda import load_csv, infer_types, types_table

loaded = load_csv("examples/sample_messy.csv")
print(types_table(infer_types(loaded.df)))
```

## Roadmap

1. Loading and column types (done)
2. Missing values, including disguised ones such as `?` and `N/A`
3. Outliers (IQR, modified z-score, Isolation Forest)
4. Duplicates, constant columns, correlated pairs, skew, class imbalance
5. Severity ranking and suggested fix code
6. HTML report and command line
7. Agent layer (LangGraph + local model) built on the tools above
8. Benchmark on datasets with injected, known problems

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
```

See `DESIGN.md` for the reasoning behind the main decisions.
