"""
Constants and enumerations for the Investment Tool.

This module defines all constant values, enumerations, and configuration
parameters used throughout the application, including:
- Data sources (Alpaca, Local)
- Execution modes (Investing, Backtesting)
- Frequencies (Daily, Weekly, Monthly)
- Stock and crypto symbols
- Model parameters and field names
"""

from __future__ import annotations

from enum import Enum, EnumMeta, unique
from typing import Any


class StringEnum(str, Enum):
    """Base enumeration class for string-valued enums.

    Inherits from both ``str`` and ``Enum`` to allow direct string comparison
    and serialization while maintaining enum semantics.
    """

    def __str__(self) -> str:
        """Return the string value of the enumeration member."""
        return str(self.value)

    @classmethod
    def list(cls) -> list[str]:
        """Return all enumeration values as a list of strings.

        Returns
        -------
        list[str]
            All enum member values.
        """
        return [str(member.value) for member in cls]


class EnumDirectValueMeta(EnumMeta):
    """Metaclass that returns enum values directly on attribute access.

    Allows accessing enum values without ``.value``, e.g., ``MyEnum.MEMBER``
    returns the value directly instead of the enum member.
    """

    def __getattribute__(cls, name: str) -> Any:
        """Return the value if accessing an enum member, otherwise the attribute.

        Parameters
        ----------
        name : str
            Attribute or enum member name.

        Returns
        -------
        Any
            The enum value or the attribute.
        """
        value = super().__getattribute__(name)
        if isinstance(value, cls):
            value = value.value
        return value


class FloatEnum(float, Enum, metaclass=EnumDirectValueMeta):
    """Base enumeration class for float-valued enums.

    Uses ``EnumDirectValueMeta`` to allow direct value access without ``.value``.
    """

    def __str__(self) -> str:
        """Return string representation of the float value."""
        return str(self.value)

    @classmethod
    def list(cls) -> list[float]:
        """Return all enumeration values as a list of floats.

        Returns
        -------
        list[float]
            All enum member values.
        """
        return [member.value for member in cls]


@unique
class DataSource(StringEnum):
    """Available data sources for financial data retrieval."""

    ALPACA = "alpaca"
    LOCAL = "local"


@unique
class ExecutionMode(StringEnum):
    """Execution modes for the investment tool."""

    INVESTING = "investing"
    BACKTESTING = "backtesting"


@unique
class Frequency(StringEnum):
    """Data frequency options for time series."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@unique
class StrategyType(StringEnum):
    """Available investment strategy types."""

    COMBINED = "combined"
    OPTIMIZATION = "optimization"
    MOMENTUM_RISK_PARITY = "momentum_risk_parity"
    BUY_AND_HOLD = "buy_and_hold"
    SENTIMENT_EWM = "sentiment_ewm"


@unique
class StockFields(StringEnum):
    """Standard field names for stock/price DataFrames.

    Used throughout the codebase to ensure consistent column naming
    for OHLCV data, technical indicators, and derived metrics.
    """

    DATE = "date"
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"
    TRADE_COUNT = "trade_count"
    VWAP = "vwap"
    RETURNS = "returns"
    LOG_RETURNS = "log_returns"
    MOVING_AVERAGE = "ma"
    VOLATILITY = "volatility"
    RSI = "rsi"
    MIDDLE_BAND = "middle_band"
    UPPER_BAND = "upper_band"
    LOWER_BAND = "lower_band"
    MACD = "macd"
    SIGNAL_LINE = "signal_line"
    SIGNAL = "signal"
    POSITIONS = "positions"
    STRATEGY_RETURNS = "strategy_returns"
    CUMULATIVE_RETURNS = "cumulative_returns"
    SENTIMENT_SCORE = "sentiment_score"


@unique
class PortfolioBalance(StringEnum):
    """Quantity bounded and diversified by the portfolio model."""

    RISK_CONTRIBUTIONS = "risk_contributions"
    CAPITAL_WEIGHTS = "capital_weights"
    HYBRID = "hybrid"


@unique
class OptimizationReturnEstimator(StringEnum):
    """Configurations of shared prediction implementations for Optimization."""

    EWM_6 = "ewm_6"
    EWM_12 = "ewm_12"
    EWM_12_MEAN_50 = "ewm_12_mean_50"
