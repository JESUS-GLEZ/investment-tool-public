"""Unit tests for custom exceptions."""

from __future__ import annotations

import pytest

from investment_tool.exceptions import (
    InvestmentToolError,
    ConfigurationError,
    MissingConfigurationError,
    DataError,
    DataRetrievalError,
    APIConnectionError,
    DataSourceNotFoundError,
    DataValidationError,
    MissingDataError,
    InvalidDataFormatError,
    DataIntegrityError,
    DataStorageError,
    StrategyError,
    OptimizationError,
    ConvergenceError,
    PredictionError,
    InsufficientDataError,
    ExecutionError,
    BacktestingError,
    PortfolioError,
    ValidationError,
)


class TestInvestmentToolError:
    """Tests for the base InvestmentToolError exception."""

    def test_basic_message(self) -> None:
        """Test exception with basic message."""
        err = InvestmentToolError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.details == {}
        assert err.original_error is None

    def test_with_details(self) -> None:
        """Test exception with additional details."""
        err = InvestmentToolError(
            "Error occurred",
            details={"code": 123, "symbol": "AAPL"},
        )
        assert "code=123" in str(err)
        assert "symbol=AAPL" in str(err)
        assert err.details == {"code": 123, "symbol": "AAPL"}

    def test_with_original_error(self) -> None:
        """Test exception chaining with original error."""
        original = ValueError("Original error")
        err = InvestmentToolError("Wrapped error", original_error=original)
        assert err.original_error is original

    def test_repr(self) -> None:
        """Test __repr__ method."""
        err = InvestmentToolError("Test", details={"key": "value"})
        repr_str = repr(err)
        assert "InvestmentToolError" in repr_str
        assert "Test" in repr_str
        assert "key" in repr_str


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_configuration_error(self) -> None:
        """Test basic ConfigurationError."""
        err = ConfigurationError("Invalid configuration")
        assert isinstance(err, InvestmentToolError)
        assert str(err) == "Invalid configuration"

    def test_missing_configuration_error(self) -> None:
        """Test MissingConfigurationError with key and section."""
        err = MissingConfigurationError("api_key", section="ALPACA")
        assert isinstance(err, ConfigurationError)
        assert "api_key" in str(err)
        assert "ALPACA" in str(err)
        assert err.details["config_key"] == "api_key"
        assert err.details["section"] == "ALPACA"

    def test_missing_configuration_without_section(self) -> None:
        """Test MissingConfigurationError without section."""
        err = MissingConfigurationError("timeout")
        assert "timeout" in str(err)
        assert err.details["section"] is None


class TestDataErrors:
    """Tests for data-related exceptions."""

    def test_data_retrieval_error(self) -> None:
        """Test DataRetrievalError."""
        err = DataRetrievalError("alpaca", "Connection timeout")
        assert isinstance(err, DataError)
        assert "alpaca" in str(err)
        assert "Connection timeout" in str(err)
        assert err.details["source"] == "alpaca"

    def test_api_connection_error(self) -> None:
        """Test APIConnectionError with status code."""
        err = APIConnectionError("Alpaca", "Rate limited", status_code=429)
        assert isinstance(err, DataRetrievalError)
        assert err.details["status_code"] == 429
        assert "Alpaca" in str(err)

    def test_data_source_not_found(self) -> None:
        """Test DataSourceNotFoundError with path."""
        err = DataSourceNotFoundError("local", path="/data/stocks.csv")
        assert isinstance(err, DataRetrievalError)
        assert "/data/stocks.csv" in str(err)
        assert err.details["path"] == "/data/stocks.csv"

    def test_data_validation_error(self) -> None:
        """Test DataValidationError with symbol and field."""
        err = DataValidationError(
            "Invalid price value",
            symbol="AAPL",
            field="close",
        )
        assert isinstance(err, DataError)
        assert err.details["symbol"] == "AAPL"
        assert err.details["field"] == "close"

    def test_missing_data_error(self) -> None:
        """Test MissingDataError."""
        err = MissingDataError("stock_dict")
        assert isinstance(err, DataValidationError)
        assert "stock_dict" in str(err)
        assert err.details["data_type"] == "stock_dict"

    def test_invalid_data_format_error(self) -> None:
        """Test InvalidDataFormatError."""
        err = InvalidDataFormatError("DataFrame", actual_format="list")
        assert isinstance(err, DataValidationError)
        assert "DataFrame" in str(err)
        assert "list" in str(err)
        assert err.details["expected_format"] == "DataFrame"
        assert err.details["actual_format"] == "list"

    def test_data_integrity_error(self) -> None:
        """Test DataIntegrityError."""
        err = DataIntegrityError("OHLC consistency")
        assert isinstance(err, DataValidationError)
        assert "OHLC consistency" in str(err)

    def test_data_storage_error(self) -> None:
        """Test DataStorageError."""
        err = DataStorageError("save", "Disk full", path="/data/output.pkl")
        assert isinstance(err, DataError)
        assert "save" in str(err)
        assert "Disk full" in str(err)
        assert err.details["path"] == "/data/output.pkl"


class TestStrategyErrors:
    """Tests for strategy-related exceptions."""

    def test_optimization_error(self) -> None:
        """Test OptimizationError."""
        err = OptimizationError("Failed to converge", optimizer="GWASFGA")
        assert isinstance(err, StrategyError)
        assert err.details["optimizer"] == "GWASFGA"

    def test_convergence_error(self) -> None:
        """Test ConvergenceError with iterations and tolerance."""
        err = ConvergenceError(iterations=1000, tolerance=1e-6)
        assert isinstance(err, OptimizationError)
        assert err.details["iterations"] == 1000
        assert err.details["tolerance"] == 1e-6

    def test_prediction_error(self) -> None:
        """Test PredictionError."""
        err = PredictionError("Model failed", model="RidgeVARX")
        assert isinstance(err, StrategyError)
        assert err.details["model"] == "RidgeVARX"

    def test_insufficient_data_error(self) -> None:
        """Test InsufficientDataError."""
        err = InsufficientDataError(required=100, actual=50)
        assert isinstance(err, StrategyError)
        assert "required 100" in str(err)
        assert "got 50" in str(err)
        assert err.details["required"] == 100
        assert err.details["actual"] == 50


class TestExecutionErrors:
    """Tests for execution-related exceptions."""

    def test_backtesting_error(self) -> None:
        """Test BacktestingError with period."""
        err = BacktestingError("Rebalance failed", period="2024-01-15")
        assert isinstance(err, ExecutionError)
        assert err.details["period"] == "2024-01-15"

    def test_portfolio_error(self) -> None:
        """Test PortfolioError with operation."""
        err = PortfolioError("Insufficient funds", operation="rebalance")
        assert isinstance(err, ExecutionError)
        assert err.details["operation"] == "rebalance"


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error_without_report(self) -> None:
        """Test ValidationError without report."""
        err = ValidationError("Validation failed")
        assert isinstance(err, InvestmentToolError)
        assert err.report is None

    def test_validation_error_with_mock_report(self) -> None:
        """Test ValidationError with a mock report."""

        class MockReport:
            def summary(self) -> str:
                return "2 errors, 3 warnings"

        report = MockReport()
        err = ValidationError("Validation failed", report=report)
        assert err.report is report
        assert err.details["has_report"] is True
        assert "2 errors" in err.details["summary"]


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_base(self) -> None:
        """Test all custom exceptions inherit from InvestmentToolError."""
        exception_classes = [
            ConfigurationError,
            MissingConfigurationError,
            DataError,
            DataRetrievalError,
            APIConnectionError,
            DataSourceNotFoundError,
            DataValidationError,
            MissingDataError,
            InvalidDataFormatError,
            DataIntegrityError,
            DataStorageError,
            StrategyError,
            OptimizationError,
            ConvergenceError,
            PredictionError,
            InsufficientDataError,
            ExecutionError,
            BacktestingError,
            PortfolioError,
            ValidationError,
        ]

        for exc_class in exception_classes:
            assert issubclass(
                exc_class, InvestmentToolError
            ), f"{exc_class.__name__} should inherit from InvestmentToolError"

    def test_catch_broad_exception(self) -> None:
        """Test catching all investment tool errors with base class."""
        errors = [
            ConfigurationError("config error"),
            DataRetrievalError("alpaca", "retrieval error"),
            OptimizationError("optimization error"),
            BacktestingError("backtest error"),
        ]

        for err in errors:
            with pytest.raises(InvestmentToolError):
                raise err

    def test_specific_catch(self) -> None:
        """Test catching specific exceptions."""
        with pytest.raises(OptimizationError):
            raise ConvergenceError("Did not converge")

        with pytest.raises(DataError):
            raise MissingDataError("prices")


class TestExceptionChaining:
    """Tests for exception chaining with original_error."""

    def test_preserve_original_error(self) -> None:
        """Test that original error is preserved in the chain."""
        original = KeyError("missing_key")

        try:
            try:
                raise original
            except KeyError as e:
                raise DataRetrievalError(
                    "local",
                    "Key not found in data",
                    original_error=e,
                ) from e
        except DataRetrievalError as err:
            assert err.original_error is original
            assert err.__cause__ is original

    def test_chain_multiple_levels(self) -> None:
        """Test exception chaining across multiple levels."""
        root_cause = IOError("File not found")

        try:
            raise DataSourceNotFoundError(
                "local",
                path="/data/missing.csv",
                original_error=root_cause,
            ) from root_cause
        except DataSourceNotFoundError as err:
            assert err.original_error is root_cause


class TestHttpStatusCodeMapping:
    """Tests for HTTP status code mapping on exceptions."""

    @pytest.mark.parametrize(
        "exc_class, expected_status",
        [
            (InvestmentToolError, 500),
            (ConfigurationError, 500),
            (MissingConfigurationError, 500),
            (DataError, 502),
            (DataRetrievalError, 502),
            (APIConnectionError, 502),
            (DataSourceNotFoundError, 404),
            (DataValidationError, 422),
            (MissingDataError, 422),
            (InvalidDataFormatError, 422),
            (DataIntegrityError, 422),
            (DataStorageError, 500),
            (StrategyError, 500),
            (OptimizationError, 500),
            (ConvergenceError, 500),
            (PredictionError, 500),
            (InsufficientDataError, 422),
            (ExecutionError, 500),
            (BacktestingError, 500),
            (PortfolioError, 500),
            (ValidationError, 422),
        ],
    )
    def test_http_status_code_class_attribute(
        self,
        exc_class: type,
        expected_status: int,
    ) -> None:
        """Verify each exception class has the correct HTTP status code."""
        assert exc_class.http_status_code == expected_status


class TestToHttpResponse:
    """Tests for the to_http_response() serialisation method."""

    def test_base_error_response(self) -> None:
        """Test to_http_response on the base exception."""
        err = InvestmentToolError("Server exploded", details={"foo": "bar"})
        resp = err.to_http_response()
        assert resp == {
            "error": "InvestmentToolError",
            "message": "Server exploded",
            "details": {"foo": "bar"},
            "status_code": 500,
        }

    def test_subclass_uses_own_status_code(self) -> None:
        """Test that subclasses embed their own status code."""
        err = DataSourceNotFoundError("local", path="/missing.csv")
        resp = err.to_http_response()
        assert resp["status_code"] == 404
        assert resp["error"] == "DataSourceNotFoundError"

    def test_validation_error_response(self) -> None:
        """Test ValidationError produces a 422 response."""
        err = ValidationError("Bad input")
        resp = err.to_http_response()
        assert resp["status_code"] == 422
        assert resp["error"] == "ValidationError"
        assert resp["message"] == "Bad input"

    def test_data_retrieval_error_response(self) -> None:
        """Test DataRetrievalError produces a 502 response."""
        err = DataRetrievalError("alpaca", "Timed out")
        resp = err.to_http_response()
        assert resp["status_code"] == 502
        assert resp["details"]["source"] == "alpaca"

    def test_response_details_are_populated(self) -> None:
        """Test that to_http_response carries through details."""
        err = InsufficientDataError(required=200, actual=10)
        resp = err.to_http_response()
        assert resp["status_code"] == 422
        assert resp["details"]["required"] == 200
        assert resp["details"]["actual"] == 10
