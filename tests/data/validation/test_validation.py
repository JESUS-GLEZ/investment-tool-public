"""Unit tests for the input data validation module."""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from investment_tool.data.validation import (
    ValidationPipeline,
    ValidationLevel,
    ValidationReport,
    ValidationIssue,
    IssueSeverity,
    ValidationError,
    RequiredColumnsValidator,
    MissingDataValidator,
    TimeSeriesConsistencyValidator,
    NumericRangeValidator,
    MinimumHistoryValidator,
    NewsDataValidator,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_stock_df():
    """Create a valid sample stock DataFrame."""
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    np.random.seed(42)  # For reproducibility
    base_price = 100
    prices = base_price + np.cumsum(np.random.randn(50) * 2)

    return pd.DataFrame(
        {
            "open": prices + np.random.uniform(-1, 1, 50),
            "high": prices + np.abs(np.random.uniform(0, 3, 50)),
            "low": prices - np.abs(np.random.uniform(0, 3, 50)),
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, 50),
        },
        index=dates,
    )


@pytest.fixture
def sample_input_data(sample_stock_df):
    """Create sample input data dictionary."""
    return {
        "stock_dict": {
            "AAPL": sample_stock_df.copy(),
            "GOOGL": sample_stock_df.copy(),
        },
        "crypto_dict": {},
        "news_dict": {},
    }


# ============================================================================
# ValidationReport Tests
# ============================================================================


class TestValidationReport:
    """Tests for ValidationReport."""

    def test_empty_report(self):
        """Empty report should have no issues."""
        report = ValidationReport()
        assert len(report.issues) == 0
        assert not report.has_errors()
        assert not report.has_critical_issues()

    def test_add_issue(self):
        """Should add issues correctly."""
        report = ValidationReport()
        issue = ValidationIssue(
            severity=IssueSeverity.WARNING,
            validator="TestValidator",
            message="Test warning",
        )
        report.add_issue(issue)
        assert len(report.issues) == 1
        assert report.issues[0].severity == IssueSeverity.WARNING

    def test_has_errors(self):
        """Should detect errors and critical issues."""
        report = ValidationReport()

        # Warning should not count as error
        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                validator="Test",
                message="Warning",
            )
        )
        assert not report.has_errors()

        # Error should count
        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                validator="Test",
                message="Error",
            )
        )
        assert report.has_errors()

    def test_has_critical_issues(self):
        """Should detect critical issues."""
        report = ValidationReport()

        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.ERROR,
                validator="Test",
                message="Error",
            )
        )
        assert not report.has_critical_issues()

        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                validator="Test",
                message="Critical",
            )
        )
        assert report.has_critical_issues()

    def test_drop_symbol(self):
        """Should track dropped symbols."""
        report = ValidationReport()
        report.drop_symbol("AAPL", "Test reason")

        assert "AAPL" in report.dropped_symbols
        assert report.dropped_symbols["AAPL"] == "Test reason"

    def test_record_modification(self):
        """Should track modifications."""
        report = ValidationReport()
        report.record_modification("AAPL", "Forward-filled 5 values")
        report.record_modification("AAPL", "Sorted index")

        assert "AAPL" in report.modified_symbols
        assert len(report.modified_symbols["AAPL"]) == 2

    def test_summary(self):
        """Summary should include key information."""
        report = ValidationReport()
        report.original_symbol_count = 10
        report.final_symbol_count = 8
        report.drop_symbol("AAPL", "Missing data")

        summary = report.summary()
        assert "10 -> 8" in summary
        assert "AAPL" in summary

    def test_to_dict(self):
        """Should convert to dictionary."""
        report = ValidationReport()
        report.add_issue(
            ValidationIssue(
                severity=IssueSeverity.WARNING,
                validator="Test",
                message="Test message",
                symbol="AAPL",
            )
        )
        report.original_symbol_count = 5
        report.final_symbol_count = 4

        result = report.to_dict()
        assert "issues" in result
        assert len(result["issues"]) == 1
        assert result["original_symbol_count"] == 5


# ============================================================================
# RequiredColumnsValidator Tests
# ============================================================================


class TestRequiredColumnsValidator:
    """Tests for RequiredColumnsValidator."""

    def test_valid_data_passes(self, sample_input_data):
        """Valid data should pass without issues."""
        validator = RequiredColumnsValidator()
        report = ValidationReport()

        result = validator.validate(sample_input_data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 0
        assert "AAPL" in result["stock_dict"]

    def test_missing_columns_detected(self):
        """Should detect missing required columns."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = {
            "stock_dict": {
                "AAPL": pd.DataFrame(
                    {
                        "open": [100] * 10,
                        "close": [101] * 10,
                        # Missing high, low, volume
                    },
                    index=dates,
                ),
            },
            "crypto_dict": {},
        }
        validator = RequiredColumnsValidator(drop_invalid=True)
        report = ValidationReport()

        result = validator.validate(data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 1
        assert "AAPL" not in result["stock_dict"]  # Should be dropped

    def test_non_dataframe_detected(self):
        """Should detect when data is not a DataFrame."""
        data = {
            "stock_dict": {
                "AAPL": [1, 2, 3],  # Not a DataFrame
            },
            "crypto_dict": {},
        }
        validator = RequiredColumnsValidator()
        report = ValidationReport()

        validator.validate(data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 1
        assert "Expected DataFrame" in errors[0].message

    def test_column_names_normalized(self):
        """Should normalize column names to lowercase."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = {
            "stock_dict": {
                "AAPL": pd.DataFrame(
                    {
                        "OPEN": [100] * 10,
                        "HIGH": [110] * 10,
                        "LOW": [90] * 10,
                        "CLOSE": [105] * 10,
                        "VOLUME": [1000000] * 10,
                    },
                    index=dates,
                ),
            },
            "crypto_dict": {},
        }
        validator = RequiredColumnsValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 0
        assert "open" in result["stock_dict"]["AAPL"].columns


# ============================================================================
# MissingDataValidator Tests
# ============================================================================


class TestMissingDataValidator:
    """Tests for MissingDataValidator."""

    def test_no_missing_data(self, sample_input_data):
        """Data without missing values should pass."""
        validator = MissingDataValidator()
        report = ValidationReport()

        result = validator.validate(sample_input_data, report)

        assert "AAPL" in result["stock_dict"]

    def test_forward_fill_applied(self):
        """Should forward-fill missing values (using past data only)."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [105] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MissingDataValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        # NaN should be filled with previous value (101)
        assert result["stock_dict"]["AAPL"]["open"].iloc[2] == 101

    def test_inf_replaced_with_nan(self):
        """Should replace Inf values with NaN then forward-fill."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, np.inf, 102, 103, 104, 105, 106, 107, 108, 109],
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [105] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MissingDataValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        # Inf should be replaced and then forward-filled
        assert not np.isinf(result["stock_dict"]["AAPL"]["open"].iloc[1])
        # Should have warning about Inf values
        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        assert any("Inf" in w.message for w in warnings)

    def test_excessive_missing_drops_symbol(self):
        """Should drop symbols exceeding missing data threshold."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [np.nan] * 5 + [100] * 5,  # 50% missing
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [np.nan] * 5 + [105] * 5,  # 50% missing
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MissingDataValidator(max_missing_pct=0.3, drop_if_exceeds=True)
        report = ValidationReport()

        result = validator.validate(data, report)

        assert "AAPL" not in result["stock_dict"]
        assert "AAPL" in report.dropped_symbols

    def test_consecutive_nan_detected(self):
        """Should detect consecutive NaN values."""
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        close_values = [100] * 5 + [np.nan] * 8 + [105] * 7  # 8 consecutive NaN
        df = pd.DataFrame(
            {
                "open": [100] * 20,
                "high": [110] * 20,
                "low": [90] * 20,
                "close": close_values,
                "volume": [1000000] * 20,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MissingDataValidator(max_consecutive_missing=5)
        report = ValidationReport()

        validator.validate(data, report)

        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        assert any("consecutive" in w.message.lower() for w in warnings)


# ============================================================================
# TimeSeriesConsistencyValidator Tests
# ============================================================================


class TestTimeSeriesConsistencyValidator:
    """Tests for TimeSeriesConsistencyValidator."""

    def test_sorted_data_passes(self, sample_input_data):
        """Sorted data should pass."""
        validator = TimeSeriesConsistencyValidator()
        report = ValidationReport()

        result = validator.validate(sample_input_data, report)

        assert "AAPL" in result["stock_dict"]

    def test_unsorted_data_sorted(self):
        """Should sort unsorted data."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102, 103, 104],
                "high": [110] * 5,
                "low": [90] * 5,
                "close": [105] * 5,
                "volume": [1000000] * 5,
            },
            index=dates[::-1],  # Reversed order
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = TimeSeriesConsistencyValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        assert result["stock_dict"]["AAPL"].index.is_monotonic_increasing
        assert "AAPL" in report.modified_symbols

    def test_duplicate_indices_removed(self):
        """Should remove duplicate timestamps."""
        dates = pd.DatetimeIndex(
            ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-04"]
        )
        df = pd.DataFrame(
            {
                "open": [100, 101, 101.5, 102, 103],
                "high": [110] * 5,
                "low": [90] * 5,
                "close": [105] * 5,
                "volume": [1000000] * 5,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = TimeSeriesConsistencyValidator(drop_duplicates=True)
        report = ValidationReport()

        result = validator.validate(data, report)

        assert not result["stock_dict"]["AAPL"].index.duplicated().any()
        assert len(result["stock_dict"]["AAPL"]) == 4

    def test_future_dates_removed(self):
        """Should remove data with future dates (prevent look-ahead bias)."""
        today = date.today()
        dates = pd.date_range(today - timedelta(days=5), periods=10, freq="D")

        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [105] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = TimeSeriesConsistencyValidator(
            check_future_dates=True,
            reference_date=today,
        )
        report = ValidationReport()

        result = validator.validate(data, report)

        # All dates should be <= today
        max_date = result["stock_dict"]["AAPL"].index.max().date()
        assert max_date <= today

        # Should have critical issue logged if future dates were found
        if any(d.date() > today for d in dates):
            critical_issues = report.get_issues_by_severity(IssueSeverity.CRITICAL)
            assert any("future dates" in str(i).lower() for i in critical_issues)


# ============================================================================
# NumericRangeValidator Tests
# ============================================================================


class TestNumericRangeValidator:
    """Tests for NumericRangeValidator."""

    def test_valid_data_passes(self, sample_input_data):
        """Valid data should pass without range issues."""
        validator = NumericRangeValidator()
        report = ValidationReport()

        validator.validate(sample_input_data, report)

        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        # Filter out OHLC consistency warnings (may occur due to random data)
        price_warnings = [
            w
            for w in warnings
            if "below minimum" in w.message or "above maximum" in w.message
        ]
        assert len(price_warnings) == 0

    def test_negative_prices_detected(self):
        """Should detect negative prices."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, -1, 102, 103, 104],  # Negative price
                "high": [110] * 5,
                "low": [90] * 5,
                "close": [105] * 5,
                "volume": [1000000] * 5,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = NumericRangeValidator()
        report = ValidationReport()

        validator.validate(data, report)

        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        assert any("below minimum" in w.message for w in warnings)

    def test_extreme_returns_detected(self):
        """Should detect extreme single-period returns."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 100, 100, 100, 100],
                "high": [110, 360, 110, 110, 110],
                "low": [90] * 5,
                "close": [100, 350, 100, 100, 100],  # 250% return
                "volume": [1000000] * 5,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = NumericRangeValidator(max_return=2.0)
        report = ValidationReport()

        validator.validate(data, report)

        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        assert any("returns >" in w.message for w in warnings)

    def test_ohlc_inconsistency_detected(self):
        """Should detect OHLC inconsistency (high < low)."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100] * 5,
                "high": [90] * 5,  # High < Low (invalid)
                "low": [110] * 5,
                "close": [100] * 5,
                "volume": [1000000] * 5,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = NumericRangeValidator()
        report = ValidationReport()

        validator.validate(data, report)

        warnings = report.get_issues_by_severity(IssueSeverity.WARNING)
        assert any("OHLC inconsistency" in w.message for w in warnings)

    def test_clip_outliers(self):
        """Should clip outliers when configured."""
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, -1, 102, 103, 104],  # Negative price
                "high": [110] * 5,
                "low": [90] * 5,
                "close": [105] * 5,
                "volume": [1000000] * 5,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = NumericRangeValidator(clip_outliers=True)
        report = ValidationReport()

        result = validator.validate(data, report)

        # Negative value should be clipped to minimum
        assert result["stock_dict"]["AAPL"]["open"].min() >= 0.0001


# ============================================================================
# MinimumHistoryValidator Tests
# ============================================================================


class TestMinimumHistoryValidator:
    """Tests for MinimumHistoryValidator."""

    def test_sufficient_history_passes(self, sample_input_data):
        """Data with sufficient history should pass."""
        validator = MinimumHistoryValidator(min_periods=30)
        report = ValidationReport()

        result = validator.validate(sample_input_data, report)

        assert "AAPL" in result["stock_dict"]

    def test_insufficient_history_dropped(self):
        """Should drop symbols with insufficient history."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [105] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MinimumHistoryValidator(min_periods=30, drop_insufficient=True)
        report = ValidationReport()

        result = validator.validate(data, report)

        assert "AAPL" not in result["stock_dict"]
        assert "AAPL" in report.dropped_symbols

    def test_error_reported_for_insufficient(self):
        """Should report error for insufficient history."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {
                "open": [100] * 10,
                "high": [110] * 10,
                "low": [90] * 10,
                "close": [105] * 10,
                "volume": [1000000] * 10,
            },
            index=dates,
        )

        data = {"stock_dict": {"AAPL": df}, "crypto_dict": {}}
        validator = MinimumHistoryValidator(min_periods=30, drop_insufficient=False)
        report = ValidationReport()

        validator.validate(data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert len(errors) == 1
        assert "10 periods" in errors[0].message


# ============================================================================
# NewsDataValidator Tests
# ============================================================================


class TestNewsDataValidator:
    """Tests for NewsDataValidator."""

    def test_empty_news_passes(self):
        """Empty news dict should pass with info message."""
        data = {"stock_dict": {}, "crypto_dict": {}, "news_dict": {}}
        validator = NewsDataValidator()
        report = ValidationReport()

        validator.validate(data, report)

        # Should have info about validated news
        infos = report.get_issues_by_severity(IssueSeverity.INFO)
        assert len(infos) >= 1

    def test_missing_news_dict(self):
        """Should handle missing news_dict gracefully."""
        data = {"stock_dict": {}, "crypto_dict": {}}
        validator = NewsDataValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        # Should not crash
        assert result is not None
        infos = report.get_issues_by_severity(IssueSeverity.INFO)
        assert any("No news_dict" in i.message for i in infos)

    def test_invalid_news_dict_type(self):
        """Should handle invalid news_dict type."""
        data = {"stock_dict": {}, "crypto_dict": {}, "news_dict": "invalid"}
        validator = NewsDataValidator()
        report = ValidationReport()

        result = validator.validate(data, report)

        errors = report.get_issues_by_severity(IssueSeverity.ERROR)
        assert any("should be dict" in e.message for e in errors)
        assert result["news_dict"] == {}


# ============================================================================
# ValidationPipeline Tests
# ============================================================================


class TestValidationPipeline:
    """Tests for ValidationPipeline."""

    def test_default_pipeline(self, sample_input_data):
        """Default pipeline should work on valid data."""
        pipeline = ValidationPipeline()

        result, report = pipeline.validate(sample_input_data)

        assert "AAPL" in result["stock_dict"]
        assert report.final_symbol_count > 0

    def test_skip_level(self, sample_input_data):
        """SKIP level should skip validation entirely."""
        pipeline = ValidationPipeline(level=ValidationLevel.SKIP)

        result, report = pipeline.validate(sample_input_data)

        assert result == sample_input_data
        assert len(report.issues) == 0

    def test_strict_level_raises(self):
        """STRICT level should raise on errors."""
        data = {
            "stock_dict": {
                "AAPL": pd.DataFrame({"invalid": [1, 2, 3]}),
            },
            "crypto_dict": {},
        }
        pipeline = ValidationPipeline(level=ValidationLevel.STRICT)

        with pytest.raises(ValidationError):
            pipeline.validate(data)

    def test_warn_level_continues(self):
        """WARN level should continue on errors."""
        data = {
            "stock_dict": {
                "AAPL": pd.DataFrame({"invalid": [1, 2, 3]}),
            },
            "crypto_dict": {},
        }
        pipeline = ValidationPipeline(level=ValidationLevel.WARN)

        result, report = pipeline.validate(data)

        # Should not raise, but report errors
        assert report.has_errors()

    def test_custom_validators(self, sample_input_data):
        """Should accept custom validator list."""
        custom_validators = [RequiredColumnsValidator()]
        pipeline = ValidationPipeline(validators=custom_validators)

        result, report = pipeline.validate(sample_input_data)

        assert len(pipeline.validators) == 1

    def test_add_validator(self):
        """Should add validators to pipeline."""
        pipeline = ValidationPipeline(validators=[])
        pipeline.add_validator(RequiredColumnsValidator())

        assert len(pipeline.validators) == 1

    def test_add_validator_at_position(self):
        """Should add validators at specific position."""
        pipeline = ValidationPipeline(
            validators=[RequiredColumnsValidator(), NumericRangeValidator()]
        )
        pipeline.add_validator(MissingDataValidator(), position=1)

        assert len(pipeline.validators) == 3
        assert pipeline.validators[1].name == "MissingDataValidator"

    def test_remove_validator(self):
        """Should remove validators by name."""
        pipeline = ValidationPipeline()
        initial_count = len(pipeline.validators)

        removed = pipeline.remove_validator("RequiredColumnsValidator")

        assert removed
        assert len(pipeline.validators) == initial_count - 1

    def test_remove_nonexistent_validator(self):
        """Should return False when removing nonexistent validator."""
        pipeline = ValidationPipeline()

        removed = pipeline.remove_validator("NonexistentValidator")

        assert not removed

    def test_symbol_counts_tracked(self, sample_input_data):
        """Should track original and final symbol counts."""
        pipeline = ValidationPipeline()

        result, report = pipeline.validate(sample_input_data)

        assert report.original_symbol_count == 2
        assert report.final_symbol_count == 2

    def test_validation_error_includes_report(self):
        """ValidationError should include the report."""
        data = {
            "stock_dict": {
                "AAPL": pd.DataFrame({"invalid": [1, 2, 3]}),
            },
            "crypto_dict": {},
        }
        pipeline = ValidationPipeline(level=ValidationLevel.STRICT)

        try:
            pipeline.validate(data)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            assert e.report is not None
            assert len(e.report.issues) > 0
