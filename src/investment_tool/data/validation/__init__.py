"""
Input data validation module.

This module provides a centralized validation layer for financial time series data
(stocks, crypto, news) to ensure data quality before it reaches investment strategies.

Key principles:
- Never fill past data with future data (avoid forward-looking bias)
- Preserve as much data as possible while ensuring consistency
- Provide detailed validation reports for transparency
"""

from .validation_report import (
    ValidationIssue,
    ValidationReport,
    IssueSeverity,
)
from .validators import (
    BaseValidator,
    RequiredColumnsValidator,
    MissingDataValidator,
    TimeSeriesConsistencyValidator,
    NumericRangeValidator,
    MinimumHistoryValidator,
    NewsDataValidator,
)
from .validation_pipeline import ValidationPipeline, ValidationLevel
from .exceptions import ValidationError

__all__ = [
    # Report
    "ValidationIssue",
    "ValidationReport",
    "IssueSeverity",
    # Validators
    "BaseValidator",
    "RequiredColumnsValidator",
    "MissingDataValidator",
    "TimeSeriesConsistencyValidator",
    "NumericRangeValidator",
    "MinimumHistoryValidator",
    "NewsDataValidator",
    # Pipeline
    "ValidationPipeline",
    "ValidationLevel",
    # Exceptions
    "ValidationError",
]
