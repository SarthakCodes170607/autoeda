"""Every threshold the checks use, in one place."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckRules:
    """Cut-off numbers for the checks. All of them can be overridden."""

    # --- missing values (share of rows)
    missing_medium: float = 0.10
    missing_high: float = 0.40
    # Above this share of missing values, suggest dropping the column instead of filling it.
    missing_drop_column: float = 0.60
    # Placeholder text such as "?" is more serious from this share of rows.
    disguised_high: float = 0.20

    # --- duplicate rows (share of rows)
    duplicate_high: float = 0.05

    # --- univariate outliers: Tukey fences, multiples of the inter-quartile range (IQR)
    iqr_mild: float = 1.5
    iqr_extreme: float = 3.0
    outlier_min_values: int = 8
    # Mild-only outliers are reported only when they are at least this share of values,
    # because about 0.7% of perfectly normal data lies outside the mild fences.
    outlier_mild_min_fraction: float = 0.03
    outlier_mild_min_count: int = 5
    # Extreme outliers at or above this share of values make the issue HIGH.
    outlier_high_fraction: float = 0.02

    # --- sentinel values: numbers used as codes for "unknown"
    sentinel_values: tuple[float, ...] = (
        -99999.0,
        -9999.0,
        -999.0,
        -99.0,
        999.0,
        9999.0,
        99999.0,
        999999.0,
    )
    sentinel_min_count: int = 3
    sentinel_min_fraction: float = 0.005

    # --- correlated columns (absolute Pearson correlation)
    corr_high: float = 0.95
    corr_low: float = 0.90
    max_corr_columns: int = 60
    max_corr_pairs: int = 20

    # --- skew
    skew_info: float = 1.0
    skew_low: float = 2.0
    skew_min_values: int = 20

    # --- categories
    imbalance_minority_low: float = 0.10
    imbalance_minority_medium: float = 0.01
    imbalance_max_classes: int = 10
    high_cardinality_unique: int = 50
    case_variants_max_unique: int = 500

    # --- missing together
    missing_together_min_fraction: float = 0.03
    missing_together_corr: float = 0.90
    missing_together_max_columns: int = 100

    # --- multivariate outliers (robust Mahalanobis distance)
    multivariate_min_rows: int = 100
    multivariate_max_columns: int = 30
    multivariate_fit_rows: int = 20_000
    multivariate_max_rows_scored: int = 200_000
    multivariate_min_unique: int = 10
    multivariate_alpha: float = 1e-4
    # If more rows than this look unusual, the data is not one compact cloud (it has clusters
    # or heavy tails), so "unusual" has no clear meaning and nothing is reported.
    multivariate_max_fraction: float = 0.02

    # --- dates written as 03/04/2024 are ambiguous when no value settles it
    default_dayfirst: bool = False

    # How many example values to keep in an issue.
    example_limit: int = 5

    # --- quality score: points removed per issue of each severity
    penalty_high: float = 12.0
    penalty_medium: float = 5.0
    penalty_low: float = 1.5
