# Design decisions

A running log of why things are the way they are. Add to it whenever you make a choice.

## Principles

1. **Library first, custom last.** Statistics come from pandas, numpy, scipy and
   scikit-learn. Custom code is only for judgment calls (what counts as a problem, how
   severe it is, what fix to suggest) and for glue.
2. **The loader is faithful.** It never cleans or converts. If it silently turned `?`
   into a missing value, the tool could no longer report `?` as a problem.
3. **Every decision explains itself.** Each column type has a `reason` string, so a
   report can say *why* a column was called an ID or a category.
4. **Thresholds live in one place.** All cut-off numbers are fields of `TypeRules`, not
   magic numbers scattered through the code.

## Module 1: loading and column types

| Decision | Why |
|---|---|
| `csv.Sniffer` for the delimiter | It is in the standard library. Limited to `, ; tab \|`. |
| Try `utf-8-sig`, then `latin-1` | `utf-8-sig` also strips Excel's byte-order mark. `latin-1` decodes every byte, so it never fails (but may show the wrong accents). |
| `low_memory=False` | pandas otherwise guesses types chunk by chunk and can disagree with itself. |
| Read one extra row when `max_rows` is set | It is the cheapest way to know whether the file really was cut short. |
| Text is tested for numbers with `pd.to_numeric` | Reuses pandas instead of writing a number parser. |
| Text is tested for dates with `pd.to_datetime(format="mixed")`, after a regex shape check | Parsing alone is too forgiving; the regex stops ordinary words being read as dates. |
| Codes with leading zeros are never read as numbers | `01234` as a number would silently lose its zero. |
| Whole numbers with very few distinct values are categorical | Things like class 1/2/3 or a 1-5 rating behave as categories. |

## Known limitations (open for decisions)

- pandas converts an all-digit column such as `00123` to the number `123` while reading.
  The leading zeros are lost before this tool sees them. A future `dtype=str` loading mode
  would fix it.
- A date like `03/04/2024` could mean 3 April or 4 March. Type detection does not care,
  but the later step that converts dates will need a `dayfirst` setting.
- European decimals (`1,5`) in a comma-separated file are not understood.
- The leading-zero, date-shape and ID rules are heuristics and can be wrong on unusual data.
  That is why every decision shows its reason.

## Coming later

- Pydantic models arrive with the issue/report objects (module 5), where validation and JSON
  output are actually needed.
