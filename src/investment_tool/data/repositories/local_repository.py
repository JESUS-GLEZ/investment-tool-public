"""
Local file data repository implementation.

This module provides the LocalRepository class for loading financial
data from local pickle or JSON files.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import TIME_CONTRACT_VERSION, validate_bar_times
from investment_tool.data.repositories._local_loader import LocalDataFetcher
from investment_tool.data.repositories.local_data_saver import LocalDataSaver
from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    DataResult,
    ExecutionQuote,
)
from investment_tool.exceptions import DataSourceNotFoundError

logger = logging.getLogger(__name__)


class LocalRepository(BaseDataRepository):
    """Repository for loading data from local files.

    Wraps the existing LocalDataFetcher to provide a clean repository
    interface. This repository loads all data from a single file rather
    than making individual requests.

    Parameters
    ----------
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    file_path : Path
        Path to the local data file (.pkl or .json).

    Attributes
    ----------
    file_path : Path
        The path to the data file.
    _cached_data : DataResult | None
        Cached data to avoid repeated file reads.
    """

    def __init__(
        self,
        frequency: Frequency,
        file_path: Path,
        symbol_identity_date: date | None = None,
    ) -> None:
        super().__init__(frequency, symbol_identity_date)
        self.file_path = file_path
        self._cached_data: DataResult | None = None
        self._logger.info(f"LocalRepository initialized with file: {file_path}")

    @property
    def source_name(self) -> str:
        """Return the data source name."""
        return "Local"

    def _load_data(self) -> DataResult:
        """Load and cache data from local file.

        Returns
        -------
        DataResult
            The loaded data.

        Raises
        ------
        DataSourceNotFoundError
            If the file does not exist.
        """
        if self._cached_data is not None:
            return self._cached_data

        self._logger.info(f"Loading data from {self.file_path}")

        try:
            raw_data = LocalDataFetcher.load_data(self.file_path)
        except FileNotFoundError as e:
            raise DataSourceNotFoundError(
                source="local",
                path=str(self.file_path),
                original_error=e,
            ) from e

        metadata = dict(raw_data.get("metadata", {}))
        if (
            self.symbol_identity_date is not None
            and metadata.get("ticker_identity_asof")
            != self.symbol_identity_date.isoformat()
        ):
            raise ValueError(
                "Dataset ticker identity date differs from the selected universe; "
                "use the matching price snapshot or fetch it again"
            )
        has_prices = any(
            not df.empty
            for key in ("stock_dict", "crypto_dict")
            for df in raw_data.get(key, {}).values()
        )
        if (
            has_prices
            and metadata.get("time_contract_version") != TIME_CONTRACT_VERSION
        ):
            raise ValueError(
                "This local dataset has no supported bar time contract. "
                "Fetch a new dataset or explicitly upgrade known legacy Alpaca data "
                "with python -m investment_tool.data.upgrade_time_data."
            )
        if metadata.get("frequency", str(self.frequency)) != str(self.frequency):
            raise ValueError("Dataset frequency differs from the requested frequency")
        prices = {
            key: {
                symbol: validate_bar_times(df, self.frequency)
                for symbol, df in raw_data.get(key, {}).items()
            }
            for key in ("stock_dict", "crypto_dict")
        }
        self._cached_data = DataResult(
            stock_dict=prices["stock_dict"],
            crypto_dict=prices["crypto_dict"],
            news_dict=raw_data.get("news_dict", {}),
            execution_dict=raw_data.get("execution_dict", {}),
            metadata={
                **metadata,
                "loaded_via": self.source_name,
                "file_path": str(self.file_path),
                "frequency": str(self.frequency),
            },
        )

        self._logger.info(
            f"Loaded {self._cached_data.symbol_count} symbols from local file"
        )
        return self._cached_data

    def clear_cache(self) -> None:
        """Clear the cached data, forcing a reload on next access."""
        self._cached_data = None
        self._logger.debug("Cache cleared")

    def fetch_stock_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Get stock data from local file.

        Note: Date filtering is not applied for local files as they contain
        pre-filtered historical data.

        Parameters
        ----------
        symbols : list[str]
            List of stock symbols (used for filtering loaded data).
        start_date : str
            Start date (ignored for local files).
        end_date : str
            End date (ignored for local files).

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping symbols to OHLCV DataFrames.
        """
        data = self._load_data()

        # Filter to requested symbols if specified
        if symbols:
            result: dict[str, pd.DataFrame] = {
                s: df for s, df in data.stock_dict.items() if s in symbols
            }
            return result
        return dict(data.stock_dict)

    def fetch_crypto_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Get crypto data from local file.

        Note: Date filtering is not applied for local files.

        Parameters
        ----------
        symbols : list[str]
            List of crypto symbols (used for filtering loaded data).
        start_date : str
            Start date (ignored for local files).
        end_date : str
            End date (ignored for local files).

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping symbols to OHLCV DataFrames.
        """
        data = self._load_data()

        if symbols:
            result: dict[str, pd.DataFrame] = {
                s: df for s, df in data.crypto_dict.items() if s in symbols
            }
            return result
        return dict(data.crypto_dict)

    def fetch_news_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[tuple[str, date], Any]:
        """Get news data from local file.

        Parameters
        ----------
        symbols : list[str]
            List of symbols (used for filtering loaded data).
        start_date : str
            Start date (ignored for local files).
        end_date : str
            End date (ignored for local files).

        Returns
        -------
        dict[tuple[str, date], Any]
            Dictionary mapping (symbol, date) tuples to news data.
        """
        data = self._load_data()
        result: dict[tuple[str, date], Any] = dict(data.news_dict)
        return result

    def fetch_all(
        self,
        stock_symbols: list[str],
        crypto_symbols: list[str],
        start_date: str,
        end_date: str,
        *,
        include_news: bool = True,
    ) -> DataResult:
        """Load all data from local file.

        Overrides the base implementation to use cached data efficiently.

        Parameters
        ----------
        stock_symbols : list[str]
            List of stock symbols (optional filtering).
        crypto_symbols : list[str]
            List of crypto symbols (optional filtering).
        start_date : str
            Start date (ignored for local files).
        end_date : str
            End date (ignored for local files).

        Returns
        -------
        DataResult
            All loaded data.
        """
        data = self._load_data()

        # The selected manifest is authoritative, including an empty crypto list.
        result_stock = {
            s: df for s, df in data.stock_dict.items() if s in stock_symbols
        }
        result_crypto = {
            s: df for s, df in data.crypto_dict.items() if s in crypto_symbols
        }

        return DataResult(
            stock_dict=result_stock,
            crypto_dict=result_crypto,
            news_dict=data.news_dict if include_news else {},
            metadata=data.metadata,
            execution_dict=data.execution_dict,
        )

    def fetch_execution_data(
        self, crypto_symbols: list[str], start_date: str, end_date: str
    ) -> dict[str, ExecutionQuote]:
        quotes: dict[str, ExecutionQuote] = self._load_data().execution_dict
        return quotes

    @staticmethod
    def save(
        data: DataResult | dict[str, Any],
        frequency: str,
        save_format: str = "pickle",
    ) -> Path:
        """Save data to local file.

        Parameters
        ----------
        data : DataResult | dict[str, Any]
            Data to save.
        frequency : str
            Data frequency for filename.
        save_format : str, optional
            Format ('json' or 'pickle'), by default 'pickle'.

        Returns
        -------
        Path
            Path to saved file.
        """
        if isinstance(data, DataResult):
            data_dict = data.to_dict()
        else:
            data_dict = data

        saved_path: Path = LocalDataSaver.save_data(data_dict, frequency, save_format)
        return saved_path
