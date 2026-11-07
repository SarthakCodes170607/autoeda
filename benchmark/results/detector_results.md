# Detector benchmark

20 trials per kind, level and base table; 6 base tables. 122 trials were skipped because the table already had that problem in the chosen column. Easy and hard levels are defined in `benchmark/detector_bench.py`.

| Problem | Level | Trials | Recall | Precision | False alarms |
|---|---|---:|---:|---:|---:|
| case_variants | easy | 120 | 100% | 100% | 0 |
| case_variants | hard | 120 | 100% | 100% | 0 |
| constant_column | easy | 120 | 100% | 100% | 0 |
| constant_column | hard | 120 | 100% | 100% | 0 |
| correlated | easy | 87 | 100% | 100% | 0 |
| correlated | hard | 87 | 100% | 100% | 0 |
| date_as_text | easy | 120 | 100% | 100% | 0 |
| date_as_text | hard | 120 | 100% | 100% | 0 |
| disguised_missing | easy | 120 | 100% | 100% | 0 |
| disguised_missing | hard | 120 | 100% | 100% | 0 |
| duplicates | easy | 120 | 100% | 100% | 0 |
| duplicates | hard | 120 | 100% | 100% | 0 |
| empty_column | easy | 120 | 100% | 100% | 0 |
| empty_column | hard | 120 | 100% | 100% | 0 |
| missing | easy | 120 | 100% | 100% | 0 |
| missing | hard | 120 | 100% | 100% | 0 |
| numeric_as_text | easy | 120 | 100% | 100% | 0 |
| numeric_as_text | hard | 120 | 100% | 100% | 0 |
| outliers | easy | 92 | 100% | 100% | 0 |
| outliers | hard | 92 | 59% | 100% | 0 |
| sentinels | easy | 120 | 100% | 100% | 0 |
| sentinels | hard | 120 | 100% | 100% | 0 |
| whitespace | easy | 120 | 100% | 100% | 0 |
| whitespace | hard | 120 | 100% | 100% | 0 |

## Unusual rows (multivariate)

| Table | Rows | Injected rows | Row recall | Flagged on clean table | False flags per 1000 rows |
|---|---:|---:|---:|---:|---:|
| iris | 150 | 100 | 34% | 0 | 0.0 |
| wine | 178 | 100 | 10% | 1 | 5.3 |
| diabetes | 442 | 100 | 98% | 3 | 2.9 |
| breast_cancer | 569 | 100 | 0% | 0 | 0.0 |
| customers | 2000 | 400 | 100% | 3 | 0.0 |
| sensors | 5000 | 1000 | 25% | 0 | 0.0 |

## Issues reported on the untouched base tables (above INFO)

- **iris** (150 rows): correlated:petal length (cm)|petal width (cm)
- **wine** (178 rows): multivariate_outliers:rows
- **diabetes** (442 rows): multivariate_outliers:rows
- **breast_cancer** (569 rows): outliers:area error, outliers:perimeter error, outliers:radius error, outliers:worst area, outliers:smoothness error, outliers:compactness error, outliers:fractal dimension error, outliers:symmetry error, outliers:mean area, outliers:worst fractal dimension, outliers:worst symmetry, outliers:concavity error, outliers:texture error, outliers:concave points error, outliers:mean compactness, outliers:worst compactness, outliers:mean fractal dimension, outliers:mean symmetry, outliers:worst perimeter, outliers:mean radius, outliers:worst concavity, outliers:mean texture, outliers:mean smoothness, correlated:mean area|worst area, correlated:mean area|worst perimeter, correlated:mean area|worst radius, correlated:mean perimeter|mean area, correlated:mean perimeter|worst perimeter, correlated:mean perimeter|worst radius, correlated:mean radius|mean area, correlated:mean radius|mean perimeter, correlated:mean radius|worst perimeter, correlated:mean radius|worst radius, correlated:radius error|area error, correlated:radius error|perimeter error, correlated:worst perimeter|worst area, correlated:worst radius|worst area, correlated:worst radius|worst perimeter, outliers:mean concavity, correlated:mean concavity|mean concave points, correlated:mean perimeter|worst area, correlated:mean radius|worst area, correlated:mean texture|worst texture, correlated:perimeter error|area error
- **customers** (2000 rows): outliers:income, outliers:spend, multivariate_outliers:rows, correlated:income|spend
- **sensors** (5000 rows): none
