"""
Base repository interface for data access.

This module defines the abstract base class for all data repositories,
providing a consistent interface for retrieving financial data.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, NotRequired, TypedDict

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import TIME_CONTRACT_VERSION, validate_bar_times

logger = logging.getLogger(__name__)


class ExecutionQuote(TypedDict):
    """Scheduled UTC stock open and crypto prices with actual UTC fill times.

    Legacy quotes without crypto_timestamps denote exact-opening fills.
    """

    # Closed daily stock sessions are explicit, never inferred from absent quotes.
    market_closed: NotRequired[bool]
    timestamp: NotRequired[str]
    crypto_prices: dict[str, float]
    crypto_timestamps: NotRequired[dict[str, str]]


@dataclass
class DataResult:
    """Container for data retrieval results.

    Provides a consistent structure for data returned by repositories,
    including metadata about the retrieval operation.

    Parameters
    ----------
    stock_dict : dict[str, pd.DataFrame]
        Dictionary mapping stock symbols to their historical OHLCV DataFrames.
    crypto_dict : dict[str, pd.DataFrame]
        Dictionary mapping crypto symbols to their historical OHLCV DataFrames.
    news_dict : dict[tuple[str, date], Any]
        Dictionary mapping (symbol, date) tuples to news data.
    metadata : dict[str, Any]
        Additional metadata about the retrieval (source, timestamps, etc.).

    Attributes
    ----------
    stock_dict : dict[str, pd.DataFrame]
    crypto_dict : dict[str, pd.DataFrame]
    news_dict : dict[tuple[str, date], Any]
    metadata : dict[str, Any]
    """

    stock_dict: dict[str, pd.DataFrame] = field(default_factory=dict)
    crypto_dict: dict[str, pd.DataFrame] = field(default_factory=dict)
    news_dict: dict[tuple[str, date], Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_dict: dict[str, ExecutionQuote] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format for backward compatibility.

        Returns
        -------
        dict[str, Any]
            Dictionary with stock_dict, crypto_dict, news_dict keys.
        """
        result: dict[str, Any] = {
            "stock_dict": self.stock_dict,
            "crypto_dict": self.crypto_dict,
            "news_dict": self.news_dict,
            "metadata": self.metadata,
        }
        if self.execution_dict:
            result["execution_dict"] = self.execution_dict
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataResult":
        """Create DataResult from dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            Dictionary with stock_dict, crypto_dict, news_dict keys.

        Returns
        -------
        DataResult
            New DataResult instance.
        """
        return cls(
            stock_dict=data.get("stock_dict", {}),
            crypto_dict=data.get("crypto_dict", {}),
            news_dict=data.get("news_dict", {}),
            execution_dict=data.get("execution_dict", {}),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_empty(self) -> bool:
        """Check if all data dictionaries are empty.

        Returns
        -------
        bool
            True if all data is empty.
        """
        return not self.stock_dict and not self.crypto_dict and not self.news_dict

    @property
    def symbol_count(self) -> int:
        """Get total number of symbols with data.

        Returns
        -------
        int
            Count of unique symbols across stocks and crypto.
        """
        return len(self.stock_dict) + len(self.crypto_dict)


class BaseDataRepository(ABC):
    """Abstract base class for data repositories.

    Defines the interface that all data repositories must implement,
    ensuring consistent access patterns regardless of data source.

    This follows the Repository Pattern, abstracting data access logic
    and providing a clean boundary between domain logic and data infrastructure.

    Parameters
    ----------
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).

    Attributes
    ----------
    frequency : Frequency
        The data frequency for this repository.
    """

    def __init__(
        self, frequency: Frequency, symbol_identity_date: date | None = None
    ) -> None:
        self.frequency = frequency
        self.symbol_identity_date = symbol_identity_date
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return the name of the data source.

        Returns
        -------
        str
            Human-readable name of the data source (e.g., 'Alpaca', 'Local').
        """
        pass

    @abstractmethod
    def fetch_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch canonical calendar-labelled bars with the shared time columns.

        Parameters
        ----------
        symbols : list[str]
            List of stock ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping symbols to their OHLCV DataFrames.
        """
        pass

    @abstractmethod
    def fetch_crypto_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical crypto data for given symbols.

        Parameters
        ----------
        symbols : list[str]
            List of crypto ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping symbols to their OHLCV DataFrames.
        """
        pass

    @abstractmethod
    def fetch_news_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[tuple[str, date], Any]:
        """Fetch news data for given symbols.

        Parameters
        ----------
        symbols : list[str]
            List of ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        dict[tuple[str, date], Any]
            Dictionary mapping (symbol, date) tuples to news data.
        """
        pass

    def fetch_execution_data(
        self, crypto_symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, ExecutionQuote]:
        """Synchronized opening quotes keyed by period-start date, separate from OHLCV."""
        return {}

    def fetch_all(
        self,
        stock_symbols: list[str],
        crypto_symbols: list[str],
        start_date: str,
        end_date: str,
        *,
        include_news: bool = True,
    ) -> DataResult:
        """Fetch all data types for given symbols.

        Convenience method that fetches stocks, crypto, and news data
        in a single call. Pass include_news=False to preflight execution
        coverage before a potentially slow news fetch.

        Parameters
        ----------
        stock_symbols : list[str]
            List of stock ticker symbols.
        crypto_symbols : list[str]
            List of crypto ticker symbols.
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        DataResult
            Container with all fetched data.
        """
        self._logger.info(
            f"Fetching all data from {self.source_name} "
            f"({len(stock_symbols)} stocks, {len(crypto_symbols)} crypto)"
        )

        stock_dict = {
            symbol: validate_bar_times(frame, self.frequency)
            for symbol, frame in self.fetch_stock_data(
                stock_symbols, start_date, end_date
            ).items()
        }
        crypto_dict = {
            symbol: validate_bar_times(frame, self.frequency)
            for symbol, frame in self.fetch_crypto_data(
                crypto_symbols, start_date, end_date
            ).items()
        }
        news_dict = (
            self.fetch_news_data(stock_symbols, start_date, end_date)
            if include_news
            else {}
        )

        result = DataResult(
            stock_dict=stock_dict,
            crypto_dict=crypto_dict,
            news_dict=news_dict,
            metadata={
                "time_contract_version": TIME_CONTRACT_VERSION,
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "availability_assumption": "calendar interval end unless the adapter supplies a later publication boundary",
                "source": self.source_name,
                "ticker_identity_asof": (
                    self.symbol_identity_date.isoformat()
                    if self.symbol_identity_date is not None
                    else None
                ),
                "frequency": str(self.frequency),
                "start_date": start_date,
                "end_date": end_date,
                "stock_count": len(stock_dict),
                "crypto_count": len(crypto_dict),
            },
        )

        self._logger.info(
            f"Fetched {result.symbol_count} symbols from {self.source_name}"
        )
        return result

    def validate_date_range(self, start_date: str, end_date: str) -> bool:
        """Validate that date range is valid.

        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format.
        end_date : str
            End date in YYYY-MM-DD format.

        Returns
        -------
        bool
            True if date range is valid.

        Raises
        ------
        ValueError
            If dates are invalid or start > end.
        """
        from datetime import datetime

        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format: {e}") from e

        if start > end:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        return True
