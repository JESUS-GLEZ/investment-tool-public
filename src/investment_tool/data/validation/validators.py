"""
Individual validators for financial time series data.

Each validator follows the Strategy pattern and can be composed in a pipeline.
Validators should:
- Never fill past data with future data (avoid forward-looking bias)
- Preserve as much data as possible
- Report all issues found
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.defaults import VALIDATION
from .validation_report import ValidationReport, ValidationIssue, IssueSeverity

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """Abstract base class for all validators.

    Attributes
    ----------
    name : str
        Human-readable name of the validator.
    """

    name: str = "BaseValidator"

    @abstractmethod
    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Validate and optionally clean the input data.

        Parameters
        ----------
        data : dict[str, Any]
            Input data dictionary with keys: stock_dict, crypto_dict, news_dict.
        report : ValidationReport
            Report to add issues to.

        Returns
        -------
        dict[str, Any]
            Validated (and possibly cleaned) data.
        """
        pass


class RequiredColumnsValidator(BaseValidator):
    """Validates that required OHLCV columns exist in stock/crypto data."""

    name = "RequiredColumnsValidator"

    # Standard OHLCV columns required for price data
    REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}

    def __init__(self, drop_invalid: bool = True) -> None:
        """
        Parameters
        ----------
        drop_invalid : bool
            If True, drop symbols missing required columns. If False, only report.
        """
        self.drop_invalid = drop_invalid

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Check that all price DataFrames have required OHLCV columns."""

        for dict_key in ("stock_dict", "crypto_dict"):
            if dict_key not in data:
                continue

            symbols_to_drop: list[str] = []

            for symbol, df in data[dict_key].items():
                if not isinstance(df, pd.DataFrame):
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            validator=self.name,
                            symbol=symbol,
                            message=f"Expected DataFrame, got {type(df).__name__}",
                        )
                    )
                    symbols_to_drop.append(symbol)
                    continue

                # Normalize column names to lowercase
                df.columns = df.columns.str.lower()

                missing_cols = self.REQUIRED_COLUMNS - set(df.columns)
                if missing_cols:
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            validator=self.name,
                            symbol=symbol,
                            message=f"Missing required columns: {missing_cols}",
                            details={"missing_columns": list(missing_cols)},
                        )
                    )
                    if self.drop_invalid:
                        symbols_to_drop.append(symbol)

            # Drop invalid symbols
            for symbol in symbols_to_drop:
                del data[dict_key][symbol]
                report.drop_symbol(symbol, "Missing required columns")

        return data


class MissingDataValidator(BaseValidator):
    """Validates and handles missing data (NaN, Inf) in time series.

    Key principle: Only use backward-looking imputation (forward-fill from past)
    to avoid forward-looking bias.
    """

    name = "MissingDataValidator"

    def __init__(
        self,
        max_missing_pct: float = VALIDATION.max_missing_pct,
        max_consecutive_missing: int = VALIDATION.max_consecutive_missing,
        fill_method: str = "ffill",
        drop_if_exceeds: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        max_missing_pct : float
            Maximum allowed percentage of missing values (0.0 to 1.0).
        max_consecutive_missing : int
            Maximum allowed consecutive missing values.
        fill_method : str
            "ffill" uses past observations; "none" preserves missing quotes.
        drop_if_exceeds : bool
            If True, drop symbols exceeding missing thresholds.
        """
        self.max_missing_pct = max_missing_pct
        self.max_consecutive_missing = max_consecutive_missing
        if fill_method not in {"ffill", "none"}:
            raise ValueError("fill_method must be ffill or none")
        self.fill_method = fill_method
        self.drop_if_exceeds = drop_if_exceeds

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Check for and handle missing values in price data."""

        for dict_key in ("stock_dict", "crypto_dict"):
            if dict_key not in data:
                continue

            symbols_to_drop: list[str] = []

            for symbol, df in list(data[dict_key].items()):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                # Check for Inf values and replace with NaN
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                inf_mask = np.isinf(df[numeric_cols])
                if inf_mask.any().any():
                    inf_count = inf_mask.sum().sum()
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            validator=self.name,
                            symbol=symbol,
                            message=f"Found {inf_count} Inf values, replaced with NaN",
                            details={"inf_count": int(inf_count)},
                        )
                    )
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    report.record_modification(symbol, "Replaced Inf with NaN")

                # Calculate missing percentage per column
                missing_pct = df.isna().mean()

                # Check if any critical column exceeds threshold
                critical_cols = ["close", "open", "high", "low"]
                exceeds_threshold = False

                for col in critical_cols:
                    if (
                        col in missing_pct.index
                        and missing_pct[col] > self.max_missing_pct
                    ):
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                validator=self.name,
                                symbol=symbol,
                                field_name=col,
                                message=(
                                    f"Missing {missing_pct[col]:.1%} > "
                                    f"threshold {self.max_missing_pct:.1%}"
                                ),
                                details={"missing_pct": float(missing_pct[col])},
                            )
                        )
                        exceeds_threshold = True

                # Check consecutive missing values
                if "close" in df.columns:
                    max_consecutive = self._max_consecutive_nan(df["close"])
                    if max_consecutive > self.max_consecutive_missing:
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                field_name="close",
                                message=(
                                    f"Max consecutive NaN: {max_consecutive} > "
                                    f"{self.max_consecutive_missing}"
                                ),
                                details={"max_consecutive_nan": max_consecutive},
                            )
                        )

                if exceeds_threshold and self.drop_if_exceeds:
                    symbols_to_drop.append(symbol)
                    continue

                # Apply forward-fill (only uses past data - no look-ahead bias)
                if self.fill_method == "ffill" and df.isna().any().any():
                    original_na_count = int(df.isna().sum().sum())
                    df.ffill(inplace=True)
                    remaining_na_count = int(df.isna().sum().sum())
                    filled_count = original_na_count - remaining_na_count

                    if filled_count > 0:
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.INFO,
                                validator=self.name,
                                symbol=symbol,
                                message=(
                                    f"Forward-filled {filled_count} missing values "
                                    "(using past data only)"
                                ),
                                details={"filled_count": filled_count},
                            )
                        )
                        report.record_modification(
                            symbol, f"Forward-filled {filled_count} values"
                        )

                    # Drop leading NaN rows (can't fill without historical data)
                    if remaining_na_count > 0:
                        # Find first row where all critical columns are non-null
                        critical_existing = [
                            c for c in critical_cols if c in df.columns
                        ]
                        if critical_existing:
                            first_valid = df[critical_existing].first_valid_index()
                            if first_valid is not None:
                                rows_to_drop = df.index.get_loc(first_valid)
                                if isinstance(rows_to_drop, int) and rows_to_drop > 0:
                                    df.drop(df.index[:rows_to_drop], inplace=True)
                                    report.add_issue(
                                        ValidationIssue(
                                            severity=IssueSeverity.WARNING,
                                            validator=self.name,
                                            symbol=symbol,
                                            message=(
                                                f"Dropped {rows_to_drop} leading rows "
                                                "with NaN (no historical data to fill)"
                                            ),
                                            details={"dropped_rows": rows_to_drop},
                                        )
                                    )
                                    report.record_modification(
                                        symbol, f"Dropped {rows_to_drop} leading rows"
                                    )

                data[dict_key][symbol] = df

            # Drop symbols exceeding thresholds
            for symbol in symbols_to_drop:
                del data[dict_key][symbol]
                report.drop_symbol(symbol, "Exceeded missing data threshold")

        return data

    @staticmethod
    def _max_consecutive_nan(series: pd.Series) -> int:
        """Calculate maximum consecutive NaN values in a series."""
        if series.isna().sum() == 0:
            return 0
        is_nan = series.isna()
        groups = (is_nan != is_nan.shift()).cumsum()
        nan_groups = is_nan.groupby(groups).sum()
        return int(nan_groups.max()) if len(nan_groups) > 0 else 0


class TimeSeriesConsistencyValidator(BaseValidator):
    """Validates time series consistency: monotonic dates, no future data, no duplicates."""

    name = "TimeSeriesConsistencyValidator"

    def __init__(
        self,
        check_future_dates: bool = True,
        reference_date: date | None = None,
        drop_duplicates: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        check_future_dates : bool
            If True, flag data points with dates in the future.
        reference_date : date | None
            Reference date to check against. Defaults to today.
        drop_duplicates : bool
            If True, drop duplicate index entries (keep first).
        """
        self.check_future_dates = check_future_dates
        self.reference_date = reference_date or date.today()
        self.drop_duplicates = drop_duplicates

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Check time series consistency."""

        for dict_key in ("stock_dict", "crypto_dict"):
            if dict_key not in data:
                continue

            for symbol, df in list(data[dict_key].items()):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            validator=self.name,
                            symbol=symbol,
                            message="Index is not DatetimeIndex, attempting conversion",
                        )
                    )
                    try:
                        df.index = pd.to_datetime(df.index)
                        report.record_modification(
                            symbol, "Converted index to DatetimeIndex"
                        )
                    except Exception as e:
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.ERROR,
                                validator=self.name,
                                symbol=symbol,
                                message=f"Failed to convert index to datetime: {e}",
                            )
                        )
                        continue

                # Check for duplicate indices
                if df.index.duplicated().any():
                    dup_count = int(df.index.duplicated().sum())
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            validator=self.name,
                            symbol=symbol,
                            message=f"Found {dup_count} duplicate timestamps",
                            details={"duplicate_count": dup_count},
                        )
                    )
                    if self.drop_duplicates:
                        df = df[~df.index.duplicated(keep="first")]
                        report.record_modification(
                            symbol, f"Removed {dup_count} duplicate rows"
                        )

                # Check monotonic (sorted) index
                if not df.index.is_monotonic_increasing:
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.WARNING,
                            validator=self.name,
                            symbol=symbol,
                            message="Index is not sorted, sorting now",
                        )
                    )
                    df = df.sort_index()
                    report.record_modification(symbol, "Sorted index")

                # Check for future dates (potential look-ahead bias)
                if self.check_future_dates:
                    # Make reference_date timezone-aware if index is timezone-aware
                    ref_datetime = pd.Timestamp(self.reference_date)
                    if (
                        isinstance(df.index, pd.DatetimeIndex)
                        and df.index.tz is not None
                    ):
                        ref_datetime = ref_datetime.tz_localize(df.index.tz)

                    future_mask = df.index > ref_datetime
                    if future_mask.any():
                        future_count = int(future_mask.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.CRITICAL,
                                validator=self.name,
                                symbol=symbol,
                                message=(
                                    f"Found {future_count} rows with future dates "
                                    "(potential look-ahead bias!)"
                                ),
                                details={
                                    "future_count": future_count,
                                    "reference_date": str(self.reference_date),
                                    "latest_date": str(df.index.max()),
                                },
                            )
                        )
                        # Remove future data to prevent look-ahead bias
                        df = df[~future_mask]
                        report.record_modification(
                            symbol, f"Removed {future_count} future-dated rows"
                        )

                # Check for gaps in the time series (informational)
                if len(df) >= 3 and isinstance(df.index, pd.DatetimeIndex):
                    expected_freq = pd.infer_freq(df.index)
                    if expected_freq is None:
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.INFO,
                                validator=self.name,
                                symbol=symbol,
                                message=(
                                    "Could not infer consistent frequency "
                                    "(may have gaps or irregular data)"
                                ),
                            )
                        )

                data[dict_key][symbol] = df

        return data


class NumericRangeValidator(BaseValidator):
    """Validates numeric values are within reasonable ranges for financial data."""

    name = "NumericRangeValidator"

    # Default ranges for financial data
    DEFAULT_RANGES: dict[str, tuple[float, float]] = {
        "open": (0.0001, 1_000_000),  # Price must be positive
        "high": (0.0001, 1_000_000),
        "low": (0.0001, 1_000_000),
        "close": (0.0001, 1_000_000),
        "volume": (0, 100_000_000_000),  # Volume can be zero
    }

    # Return thresholds (single-period returns)
    MAX_SINGLE_PERIOD_RETURN = 2.0  # 200% gain in one period is suspicious
    MIN_SINGLE_PERIOD_RETURN = -0.99  # Can't lose more than 99%

    def __init__(
        self,
        custom_ranges: dict[str, tuple[float, float]] | None = None,
        clip_outliers: bool = False,
        max_return: float | None = None,
        min_return: float | None = None,
    ) -> None:
        """
        Parameters
        ----------
        custom_ranges : dict | None
            Custom (min, max) ranges per column.
        clip_outliers : bool
            If True, clip values to valid ranges instead of just flagging.
        max_return : float | None
            Maximum allowed single-period return.
        min_return : float | None
            Minimum allowed single-period return.
        """
        self.ranges = {**self.DEFAULT_RANGES, **(custom_ranges or {})}
        self.clip_outliers = clip_outliers
        self.max_return = max_return or self.MAX_SINGLE_PERIOD_RETURN
        self.min_return = min_return or self.MIN_SINGLE_PERIOD_RETURN

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Check that numeric values are within reasonable ranges."""

        for dict_key in ("stock_dict", "crypto_dict"):
            if dict_key not in data:
                continue

            for symbol, df in list(data[dict_key].items()):
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue

                # Check OHLCV columns
                for col, (min_val, max_val) in self.ranges.items():
                    if col not in df.columns:
                        continue

                    below_min = df[col] < min_val
                    above_max = df[col] > max_val

                    if below_min.any():
                        count = int(below_min.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                field_name=col,
                                message=f"{count} values below minimum {min_val}",
                                details={
                                    "count": count,
                                    "min_value": float(df[col].min()),
                                },
                            )
                        )
                        if self.clip_outliers:
                            # Convert column to float to avoid dtype incompatibility
                            df[col] = df[col].astype(float)
                            df.loc[below_min, col] = min_val
                            report.record_modification(
                                symbol, f"Clipped {count} low values in {col}"
                            )

                    if above_max.any():
                        count = int(above_max.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                field_name=col,
                                message=f"{count} values above maximum {max_val}",
                                details={
                                    "count": count,
                                    "max_value": float(df[col].max()),
                                },
                            )
                        )
                        if self.clip_outliers:
                            # Convert column to float to avoid dtype incompatibility
                            df[col] = df[col].astype(float)
                            df.loc[above_max, col] = max_val
                            report.record_modification(
                                symbol, f"Clipped {count} high values in {col}"
                            )

                # Check OHLC consistency (High >= Low, etc.)
                if all(c in df.columns for c in ["open", "high", "low", "close"]):
                    inconsistent = (
                        (df["high"] < df["low"])
                        | (df["high"] < df["open"])
                        | (df["high"] < df["close"])
                        | (df["low"] > df["open"])
                        | (df["low"] > df["close"])
                    )
                    if inconsistent.any():
                        count = int(inconsistent.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                message=(
                                    f"{count} rows with OHLC inconsistency "
                                    "(e.g., high < low)"
                                ),
                                details={"inconsistent_count": count},
                            )
                        )

                # Check for extreme single-period returns
                if "close" in df.columns and len(df) > 1:
                    returns = df["close"].pct_change(fill_method=None)
                    extreme_high = returns > self.max_return
                    extreme_low = returns < self.min_return

                    if extreme_high.any():
                        count = int(extreme_high.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                field_name="returns",
                                message=(
                                    f"{count} periods with returns > "
                                    f"{self.max_return:.0%}"
                                ),
                                details={
                                    "count": count,
                                    "max_return": float(returns.max()),
                                },
                            )
                        )

                    if extreme_low.any():
                        count = int(extreme_low.sum())
                        report.add_issue(
                            ValidationIssue(
                                severity=IssueSeverity.WARNING,
                                validator=self.name,
                                symbol=symbol,
                                field_name="returns",
                                message=(
                                    f"{count} periods with returns < "
                                    f"{self.min_return:.0%}"
                                ),
                                details={
                                    "count": count,
                                    "min_return": float(returns.min()),
                                },
                            )
                        )

                data[dict_key][symbol] = df

        return data


class MinimumHistoryValidator(BaseValidator):
    """Validates that symbols have minimum required historical data."""

    name = "MinimumHistoryValidator"

    def __init__(
        self,
        min_periods: int = VALIDATION.min_periods,
        drop_insufficient: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        min_periods : int
            Minimum number of data points required.
        drop_insufficient : bool
            If True, drop symbols with insufficient history.
        """
        self.min_periods = min_periods
        self.drop_insufficient = drop_insufficient

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Check that symbols have enough historical data."""

        for dict_key in ("stock_dict", "crypto_dict"):
            if dict_key not in data:
                continue

            symbols_to_drop: list[str] = []

            for symbol, df in data[dict_key].items():
                if not isinstance(df, pd.DataFrame):
                    continue

                if len(df) < self.min_periods:
                    report.add_issue(
                        ValidationIssue(
                            severity=IssueSeverity.ERROR,
                            validator=self.name,
                            symbol=symbol,
                            message=(
                                f"Only {len(df)} periods, "
                                f"minimum required is {self.min_periods}"
                            ),
                            details={
                                "actual_periods": len(df),
                                "min_required": self.min_periods,
                            },
                        )
                    )
                    if self.drop_insufficient:
                        symbols_to_drop.append(symbol)

            for symbol in symbols_to_drop:
                del data[dict_key][symbol]
                report.drop_symbol(
                    symbol,
                    f"Insufficient history ({self.min_periods} periods required)",
                )

        return data


class NewsDataValidator(BaseValidator):
    """Validates news data structure and content."""

    name = "NewsDataValidator"

    def __init__(
        self,
        check_future_dates: bool = True,
        reference_date: date | None = None,
    ) -> None:
        """
        Parameters
        ----------
        check_future_dates : bool
            If True, flag news with future dates.
        reference_date : date | None
            Reference date for future date check.
        """
        self.check_future_dates = check_future_dates
        self.reference_date = reference_date or date.today()

    def validate(
        self,
        data: dict[str, Any],
        report: ValidationReport,
    ) -> dict[str, Any]:
        """Validate news data structure."""

        if "news_dict" not in data:
            report.add_issue(
                ValidationIssue(
                    severity=IssueSeverity.INFO,
                    validator=self.name,
                    message="No news_dict in input data",
                )
            )
            return data

        news_dict = data["news_dict"]

        if not isinstance(news_dict, dict):
            report.add_issue(
                ValidationIssue(
                    severity=IssueSeverity.ERROR,
                    validator=self.name,
                    message=f"news_dict should be dict, got {type(news_dict).__name__}",
                )
            )
            data["news_dict"] = {}
            return data

        total_news_items = 0
        future_news_count = 0

        for symbol, news_items in news_dict.items():
            if not isinstance(news_items, dict):
                report.add_issue(
                    ValidationIssue(
                        severity=IssueSeverity.WARNING,
                        validator=self.name,
                        symbol=symbol,
                        message=(
                            f"News items should be dict, "
                            f"got {type(news_items).__name__}"
                        ),
                    )
                )
                continue

            for period_key, news_data in news_items.items():
                total_news_items += 1

                # Check for future-dated news (potential look-ahead bias)
                if (
                    self.check_future_dates
                    and isinstance(period_key, tuple)
                    and len(period_key) > 0
                ):
                    try:
                        period_date = pd.to_datetime(period_key[0]).date()
                        if period_date > self.reference_date:
                            future_news_count += 1
                    except Exception:
                        pass

        if future_news_count > 0:
            report.add_issue(
                ValidationIssue(
                    severity=IssueSeverity.CRITICAL,
                    validator=self.name,
                    message=(
                        f"Found {future_news_count} news items with future dates "
                        "(look-ahead bias!)"
                    ),
                    details={"future_news_count": future_news_count},
                )
            )

        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.INFO,
                validator=self.name,
                message=(
                    f"Validated {total_news_items} news items "
                    f"across {len(news_dict)} symbols"
                ),
                details={
                    "total_news": total_news_items,
                    "symbols_with_news": len(news_dict),
                },
            )
        )

        return data
