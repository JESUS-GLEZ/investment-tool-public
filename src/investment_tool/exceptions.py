"""
Custom exceptions for the investment tool.

This module defines a comprehensive exception hierarchy for better error
handling, debugging, and user feedback throughout the application.

Exception Hierarchy
-------------------
InvestmentToolError (base)
├── ConfigurationError
│   └── MissingConfigurationError
├── DataError
│   ├── DataRetrievalError
│   │   ├── APIConnectionError
│   │   └── DataSourceNotFoundError
│   ├── DataValidationError
│   │   ├── MissingDataError
│   │   ├── InvalidDataFormatError
│   │   └── DataIntegrityError
│   └── DataStorageError
├── StrategyError
│   ├── OptimizationError
│   │   └── ConvergenceError
│   ├── PredictionError
│   └── InsufficientDataError
├── ExecutionError
│   ├── BacktestingError
│   └── PortfolioError
└── ValidationError
"""

from __future__ import annotations

from typing import Any


class InvestmentToolError(Exception):
    """Base exception for all investment tool errors.

    All custom exceptions in this application inherit from this class,
    allowing for broad exception catching when needed.

    Parameters
    ----------
    message : str
        Human-readable error description.
    details : dict[str, Any] | None, optional
        Additional context about the error.
    original_error : Exception | None, optional
        The original exception that caused this error.

    Attributes
    ----------
    message : str
    details : dict[str, Any]
    original_error : Exception | None
    http_status_code : int
        HTTP status code for API responses. Defaults to 500.

    Examples
    --------
    >>> raise InvestmentToolError("Something went wrong", details={"code": 123})
    """

    http_status_code: int = 500

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        self.original_error = original_error
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return formatted error message with details if present."""
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"details={self.details!r}, "
            f"original_error={self.original_error!r})"
        )

    def to_http_response(self) -> dict[str, Any]:
        """Serialise to a dict suitable for an HTTP error response body.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``error``, ``message``, ``details``, and
            ``status_code`` keys.
        """
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
            "status_code": self.http_status_code,
        }


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(InvestmentToolError):
    """Error related to application configuration.

    Raised when there are issues with configuration files, environment
    variables, or parameter validation.
    """

    pass


class MissingConfigurationError(ConfigurationError):
    """Required configuration value is missing.

    Parameters
    ----------
    config_key : str
        The missing configuration key.
    section : str | None, optional
        The configuration section where the key was expected.
    """

    def __init__(
        self,
        config_key: str,
        section: str | None = None,
        **kwargs: Any,
    ) -> None:
        section_info = f" in section '{section}'" if section else ""
        message = f"Missing required configuration: '{config_key}'{section_info}"
        details = {"config_key": config_key, "section": section}
        super().__init__(message, details=details, **kwargs)


# =============================================================================
# Data Errors
# =============================================================================


class DataError(InvestmentToolError):
    """Base class for data-related errors.

    Covers errors during data retrieval, validation, and storage.
    """

    http_status_code: int = 502


class DataRetrievalError(DataError):
    """Error occurred while retrieving data from a source.

    Parameters
    ----------
    source : str
        The data source that failed (e.g., 'alpaca', 'local').
    message : str
        Description of the retrieval failure.
    """

    def __init__(
        self,
        source: str,
        message: str,
        **kwargs: Any,
    ) -> None:
        full_message = f"Failed to retrieve data from '{source}': {message}"
        details = kwargs.pop("details", {})
        details["source"] = source
        super().__init__(full_message, details=details, **kwargs)


class APIConnectionError(DataRetrievalError):
    """Failed to connect to an external API.

    Parameters
    ----------
    api_name : str
        Name of the API (e.g., 'Alpaca', 'Alpha Vantage').
    status_code : int | None, optional
        HTTP status code if available.
    """

    def __init__(
        self,
        api_name: str,
        message: str = "Connection failed",
        status_code: int | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if status_code:
            details["status_code"] = status_code
        super().__init__(source=api_name, message=message, details=details, **kwargs)


class DataSourceNotFoundError(DataRetrievalError):
    """Requested data source does not exist or is unavailable.

    Parameters
    ----------
    source : str
        The data source identifier.
    path : str | None, optional
        File path if applicable.
    """

    http_status_code: int = 404

    def __init__(
        self,
        source: str,
        path: str | None = None,
        **kwargs: Any,
    ) -> None:
        message = "Data source not found"
        if path:
            message += f" at path: {path}"
        details = kwargs.pop("details", {})
        if path:
            details["path"] = path
        super().__init__(source=source, message=message, details=details, **kwargs)


class DataValidationError(DataError):
    """Data failed validation checks.

    Parameters
    ----------
    message : str
        Description of the validation failure.
    symbol : str | None, optional
        The symbol that failed validation.
    field : str | None, optional
        The specific field that failed.
    """

    http_status_code: int = 422

    def __init__(
        self,
        message: str,
        symbol: str | None = None,
        field: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if symbol:
            details["symbol"] = symbol
        if field:
            details["field"] = field
        super().__init__(message, details=details, **kwargs)


class MissingDataError(DataValidationError):
    """Required data is missing or incomplete.

    Parameters
    ----------
    data_type : str
        Type of missing data (e.g., 'stock_dict', 'prices').
    """

    def __init__(
        self,
        data_type: str,
        message: str | None = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Missing required data: {data_type}"
        details = kwargs.pop("details", {})
        details["data_type"] = data_type
        super().__init__(msg, details=details, **kwargs)


class InvalidDataFormatError(DataValidationError):
    """Data format is invalid or unexpected.

    Parameters
    ----------
    expected_format : str
        The expected data format.
    actual_format : str | None, optional
        The actual format received.
    """

    def __init__(
        self,
        expected_format: str,
        actual_format: str | None = None,
        message: str | None = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Invalid data format. Expected: {expected_format}"
        if actual_format:
            msg += f", got: {actual_format}"
        details = kwargs.pop("details", {})
        details["expected_format"] = expected_format
        if actual_format:
            details["actual_format"] = actual_format
        super().__init__(msg, details=details, **kwargs)


class DataIntegrityError(DataValidationError):
    """Data integrity check failed (e.g., OHLC inconsistency).

    Parameters
    ----------
    check_name : str
        Name of the integrity check that failed.
    """

    def __init__(
        self,
        check_name: str,
        message: str | None = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Data integrity check failed: {check_name}"
        details = kwargs.pop("details", {})
        details["check_name"] = check_name
        super().__init__(msg, details=details, **kwargs)


class DataStorageError(DataError):
    """Error occurred while storing or saving data.

    Parameters
    ----------
    operation : str
        The storage operation that failed (e.g., 'save', 'write').
    path : str | None, optional
        The file path involved.
    """

    http_status_code: int = 500

    def __init__(
        self,
        operation: str,
        message: str,
        path: str | None = None,
        **kwargs: Any,
    ) -> None:
        full_message = f"Data storage error during '{operation}': {message}"
        details = kwargs.pop("details", {})
        details["operation"] = operation
        if path:
            details["path"] = path
        super().__init__(full_message, details=details, **kwargs)


# =============================================================================
# Strategy Errors
# =============================================================================


class StrategyError(InvestmentToolError):
    """Base class for strategy-related errors.

    Covers errors in optimization, prediction, and strategy execution.
    """

    pass


class OptimizationError(StrategyError):
    """Portfolio optimization failed.

    Parameters
    ----------
    optimizer : str | None, optional
        Name of the optimizer that failed.
    """

    def __init__(
        self,
        message: str,
        optimizer: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if optimizer:
            details["optimizer"] = optimizer
        super().__init__(message, details=details, **kwargs)


class ConvergenceError(OptimizationError):
    """Optimization algorithm failed to converge.

    Parameters
    ----------
    iterations : int | None, optional
        Number of iterations completed before failure.
    tolerance : float | None, optional
        The convergence tolerance that was not met.
    """

    def __init__(
        self,
        message: str = "Optimization failed to converge",
        iterations: int | None = None,
        tolerance: float | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if iterations is not None:
            details["iterations"] = iterations
        if tolerance is not None:
            details["tolerance"] = tolerance
        super().__init__(message, details=details, **kwargs)


class PredictionError(StrategyError):
    """Return prediction failed.

    Parameters
    ----------
    model : str | None, optional
        Name of the prediction model that failed.
    """

    def __init__(
        self,
        message: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if model:
            details["model"] = model
        super().__init__(message, details=details, **kwargs)


class InsufficientDataError(StrategyError):
    """Not enough data to execute the strategy.

    Parameters
    ----------
    required : int
        Minimum required data points.
    actual : int
        Actual data points available.
    """

    http_status_code: int = 422

    def __init__(
        self,
        required: int,
        actual: int,
        message: str | None = None,
        **kwargs: Any,
    ) -> None:
        msg = message or f"Insufficient data: required {required}, got {actual}"
        details = kwargs.pop("details", {})
        details["required"] = required
        details["actual"] = actual
        super().__init__(msg, details=details, **kwargs)


# =============================================================================
# Execution Errors
# =============================================================================


class ExecutionError(InvestmentToolError):
    """Base class for execution-related errors.

    Covers errors during backtesting and live execution.
    """

    pass


class BacktestingError(ExecutionError):
    """Error occurred during backtesting.

    Parameters
    ----------
    period : str | None, optional
        The trading period where the error occurred.
    """

    def __init__(
        self,
        message: str,
        period: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if period:
            details["period"] = period
        super().__init__(message, details=details, **kwargs)


class PortfolioError(ExecutionError):
    """Error related to portfolio operations.

    Parameters
    ----------
    operation : str | None, optional
        The portfolio operation that failed (e.g., 'rebalance', 'update').
    """

    def __init__(
        self,
        message: str,
        operation: str | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        if operation:
            details["operation"] = operation
        super().__init__(message, details=details, **kwargs)


class BacktestReadinessError(BacktestingError):
    """A full-path audit blocked execution; the report remains serializable."""

    def __init__(self, report: dict[str, Any]) -> None:
        self.validation_report = report
        symbols = report["readiness"]["affected_symbols"]
        super().__init__(
            "Backtest blocked before optimization: unresolved lifecycle/calendar coverage. "
            f"Affected assets: {len(symbols)} ({', '.join(symbols[:10])}). "
            "Review data_validation_report.json for every dated gap. Supply compatible "
            "quotes or sourced corporate actions; no historical assets were deleted."
        )


# =============================================================================
# Validation Errors (from validation pipeline)
# =============================================================================


class ValidationError(InvestmentToolError):
    """Validation pipeline error with attached report.

    This exception is raised when validation fails in STRICT mode and
    includes the full validation report.

    Parameters
    ----------
    message : str
        Error description.
    report : Any | None, optional
        The ValidationReport object with detailed issues.
    """

    http_status_code: int = 422

    def __init__(
        self,
        message: str,
        report: Any | None = None,
        **kwargs: Any,
    ) -> None:
        details = kwargs.pop("details", {})
        self.report = report
        if report is not None:
            details["has_report"] = True
            if hasattr(report, "summary"):
                details["summary"] = report.summary()
        super().__init__(message, details=details, **kwargs)
