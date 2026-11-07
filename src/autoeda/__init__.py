"""autoeda: automatic exploratory data analysis with suggested fixes."""

from autoeda.drift import ColumnDrift, DriftReport, compare
from autoeda.fixes import CleanResult, apply_fix, clean, fix_to_code
from autoeda.loader import LoadedDataset, load_csv
from autoeda.models import (
    ColumnProfile,
    CorrelationMatrix,
    Fix,
    FixAction,
    Issue,
    IssueKind,
    Report,
    Severity,
)
from autoeda.profile import profile, quality_score
from autoeda.rules import CheckRules
from autoeda.types import (
    ColumnKind,
    ColumnType,
    TypeRules,
    infer_column_type,
    infer_types,
    types_table,
)

__version__ = "0.1.1"

__all__ = [
    "CheckRules",
    "CleanResult",
    "ColumnDrift",
    "ColumnKind",
    "ColumnProfile",
    "ColumnType",
    "CorrelationMatrix",
    "DriftReport",
    "Fix",
    "FixAction",
    "Issue",
    "IssueKind",
    "LoadedDataset",
    "Report",
    "Severity",
    "TypeRules",
    "__version__",
    "apply_fix",
    "clean",
    "compare",
    "fix_to_code",
    "infer_column_type",
    "infer_types",
    "load_csv",
    "profile",
    "quality_score",
    "types_table",
]
