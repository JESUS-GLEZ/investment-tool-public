"""
Investment Tool - A comprehensive portfolio optimization and backtesting framework.

This package provides tools for:
- Data retrieval from multiple sources (Alpaca, local files)
- Data validation with configurable strictness
- Portfolio optimization using multi-objective algorithms
- Backtesting with transaction costs
- Strategy development and evaluation
"""

from investment_tool.exceptions import (
    # Base exception
    InvestmentToolError,
    # Configuration errors
    ConfigurationError,
    MissingConfigurationError,
    # Data errors
    DataError,
    DataRetrievalError,
    APIConnectionError,
    DataSourceNotFoundError,
    DataValidationError,
    MissingDataError,
    InvalidDataFormatError,
    DataIntegrityError,
    DataStorageError,
    # Strategy errors
    StrategyError,
    OptimizationError,
    ConvergenceError,
    PredictionError,
    InsufficientDataError,
    # Execution errors
    ExecutionError,
    BacktestingError,
    PortfolioError,
    # Validation errors
    ValidationError,
)

__all__ = [
    # Base exception
    "InvestmentToolError",
    # Configuration errors
    "ConfigurationError",
    "MissingConfigurationError",
    # Data errors
    "DataError",
    "DataRetrievalError",
    "APIConnectionError",
    "DataSourceNotFoundError",
    "DataValidationError",
    "MissingDataError",
    "InvalidDataFormatError",
    "DataIntegrityError",
    "DataStorageError",
    # Strategy errors
    "StrategyError",
    "OptimizationError",
    "ConvergenceError",
    "PredictionError",
    "InsufficientDataError",
    # Execution errors
    "ExecutionError",
    "BacktestingError",
    "PortfolioError",
    # Validation errors
    "ValidationError",
]

__version__ = "0.1.0"
