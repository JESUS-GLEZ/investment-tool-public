"""Unit tests for Repository pattern implementation."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from investment_tool.domain.market_time import normalize_bars
from investment_tool.constants import DataSource, Frequency
from investment_tool.data.repositories.base_repository import (
    BaseDataRepository,
    DataResult,
)
from investment_tool.data.repositories.local_repository import LocalRepository
from investment_tool.data.repositories.repository_factory import (
    RepositoryFactory,
    get_repository,
)
from investment_tool.exceptions import ConfigurationError, DataSourceNotFoundError


def _monthly_price(price):
    return normalize_bars(
        pd.DataFrame(
            {"close": [price]}, index=pd.DatetimeIndex(["2023-01-01"], tz="UTC")
        ),
        Frequency.MONTHLY,
        market_timezone="UTC",
    )


class TestDataResult:
    """Tests for DataResult dataclass."""

    def test_empty_data_result(self) -> None:
        """Test creating empty DataResult."""
        result = DataResult()
        assert result.stock_dict == {}
        assert result.crypto_dict == {}
        assert result.news_dict == {}
        assert result.metadata == {}
        assert result.is_empty
        assert result.symbol_count == 0

    def test_data_result_with_data(self) -> None:
        """Test DataResult with actual data."""
        stock_dict = {"AAPL": pd.DataFrame({"close": [100, 101]})}
        crypto_dict = {"BTC/USD": pd.DataFrame({"close": [50000, 51000]})}

        result = DataResult(
            stock_dict=stock_dict,
            crypto_dict=crypto_dict,
            metadata={"source": "test"},
        )

        assert not result.is_empty
        assert result.symbol_count == 2
        assert "AAPL" in result.stock_dict
        assert result.metadata["source"] == "test"

    def test_to_dict(self) -> None:
        """Test DataResult.to_dict() method."""
        result = DataResult(
            stock_dict={"AAPL": pd.DataFrame()},
            crypto_dict={"BTC/USD": pd.DataFrame()},
            news_dict={(("AAPL", date(2024, 1, 1))): []},
        )

        d = result.to_dict()

        assert "stock_dict" in d
        assert "crypto_dict" in d
        assert "news_dict" in d
        assert "AAPL" in d["stock_dict"]

    def test_from_dict(self) -> None:
        """Test DataResult.from_dict() classmethod."""
        input_dict = {
            "stock_dict": {"AAPL": pd.DataFrame()},
            "crypto_dict": {},
            "news_dict": {},
            "metadata": {"time_contract_version": 1, "frequency": "monthly"},
        }

        result = DataResult.from_dict(input_dict)

        assert "AAPL" in result.stock_dict
        assert result.crypto_dict == {}

    def test_from_dict_missing_keys(self) -> None:
        """Test from_dict handles missing keys."""
        result = DataResult.from_dict({})

        assert result.stock_dict == {}
        assert result.crypto_dict == {}
        assert result.news_dict == {}


class TestBaseDataRepository:
    """Tests for BaseDataRepository abstract class."""

    def test_validate_date_range_valid(self) -> None:
        """Test valid date range passes validation."""

        # Create a concrete implementation for testing
        class ConcreteRepo(BaseDataRepository):
            @property
            def source_name(self) -> str:
                return "Test"

            def fetch_stock_data(self, symbols, start, end):
                return {}

            def fetch_crypto_data(self, symbols, start, end):
                return {}

            def fetch_news_data(self, symbols, start, end):
                return {}

        repo = ConcreteRepo(Frequency.DAILY)
        assert repo.validate_date_range("2024-01-01", "2024-12-31")

    def test_validate_date_range_invalid_format(self) -> None:
        """Test invalid date format raises ValueError."""

        class ConcreteRepo(BaseDataRepository):
            @property
            def source_name(self) -> str:
                return "Test"

            def fetch_stock_data(self, symbols, start, end):
                return {}

            def fetch_crypto_data(self, symbols, start, end):
                return {}

            def fetch_news_data(self, symbols, start, end):
                return {}

        repo = ConcreteRepo(Frequency.DAILY)

        with pytest.raises(ValueError, match="Invalid date format"):
            repo.validate_date_range("01-01-2024", "2024-12-31")

    def test_validate_date_range_start_after_end(self) -> None:
        """Test start date after end date raises ValueError."""

        class ConcreteRepo(BaseDataRepository):
            @property
            def source_name(self) -> str:
                return "Test"

            def fetch_stock_data(self, symbols, start, end):
                return {}

            def fetch_crypto_data(self, symbols, start, end):
                return {}

            def fetch_news_data(self, symbols, start, end):
                return {}

        repo = ConcreteRepo(Frequency.DAILY)

        with pytest.raises(ValueError, match="after end date"):
            repo.validate_date_range("2024-12-31", "2024-01-01")


class TestLocalRepository:
    """Tests for LocalRepository."""

    def test_source_name(self) -> None:
        """Test source_name property."""
        repo = LocalRepository(
            frequency=Frequency.MONTHLY,
            file_path=Path("/fake/path.pkl"),
        )
        assert repo.source_name == "Local"

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        repo = LocalRepository(
            frequency=Frequency.MONTHLY,
            file_path=Path("/fake/path.pkl"),
        )

        # Set some cached data
        repo._cached_data = DataResult(stock_dict={"AAPL": pd.DataFrame()})

        repo.clear_cache()

        assert repo._cached_data is None

    @patch("investment_tool.data.repositories.local_repository.LocalDataFetcher")
    def test_load_data_caching(self, mock_fetcher: MagicMock) -> None:
        """Test that data is cached after first load."""
        mock_fetcher.load_data.return_value = {
            "stock_dict": {"AAPL": pd.DataFrame()},
            "crypto_dict": {},
            "news_dict": {},
            "metadata": {"time_contract_version": 1, "frequency": "monthly"},
        }

        repo = LocalRepository(
            frequency=Frequency.MONTHLY,
            file_path=Path("/fake/path.pkl"),
        )

        # First call should load
        repo._load_data()
        # Second call should use cache
        repo._load_data()

        # Should only call load_data once
        assert mock_fetcher.load_data.call_count == 1

    @patch("investment_tool.data.repositories.local_repository.LocalDataFetcher")
    def test_fetch_stock_data_filters_symbols(self, mock_fetcher: MagicMock) -> None:
        """Test fetch_stock_data filters to requested symbols."""
        mock_fetcher.load_data.return_value = {
            "stock_dict": {
                "AAPL": _monthly_price(100),
                "MSFT": _monthly_price(200),
                "GOOG": _monthly_price(300),
            },
            "crypto_dict": {},
            "news_dict": {},
            "metadata": {"time_contract_version": 1, "frequency": "monthly"},
        }

        repo = LocalRepository(
            frequency=Frequency.MONTHLY,
            file_path=Path("/fake/path.pkl"),
        )

        result = repo.fetch_stock_data(["AAPL", "MSFT"], "2024-01-01", "2024-12-31")

        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" not in result

    def test_file_not_found_raises_error(self) -> None:
        """Test FileNotFoundError is wrapped in DataSourceNotFoundError."""
        repo = LocalRepository(
            frequency=Frequency.MONTHLY,
            file_path=Path("/nonexistent/path.pkl"),
        )

        with pytest.raises(DataSourceNotFoundError):
            repo._load_data()


class TestRepositoryFactory:
    """Tests for RepositoryFactory."""

    def test_get_supported_sources(self) -> None:
        """Test listing supported data sources."""
        sources = RepositoryFactory.get_supported_sources()

        assert DataSource.ALPACA in sources
        assert DataSource.LOCAL in sources

    def test_create_local_repository(self) -> None:
        """Test creating LocalRepository."""
        repo = RepositoryFactory.create(
            data_source=DataSource.LOCAL,
            frequency=Frequency.MONTHLY,
            file_path=Path("/fake/path.pkl"),
        )

        assert isinstance(repo, LocalRepository)
        assert repo.frequency == Frequency.MONTHLY

    def test_create_local_without_path_raises(self) -> None:
        """Test LOCAL without file_path raises ConfigurationError."""
        with pytest.raises(ConfigurationError, match="file_path is required"):
            RepositoryFactory.create(
                data_source=DataSource.LOCAL,
                frequency=Frequency.MONTHLY,
            )

    def test_create_unsupported_source_raises(self) -> None:
        """Test unsupported data source raises ConfigurationError."""
        # Create a fake DataSource
        with pytest.raises(ConfigurationError, match="Unsupported data source"):
            RepositoryFactory.create(
                data_source="invalid",  # type: ignore
                frequency=Frequency.MONTHLY,
            )


class TestGetRepository:
    """Tests for get_repository convenience function."""

    def test_creates_repository(self) -> None:
        """Test get_repository creates correct repository type."""
        repo = get_repository(
            data_source=DataSource.LOCAL,
            frequency=Frequency.DAILY,
            file_path=Path("/fake/path.pkl"),
        )

        assert isinstance(repo, LocalRepository)

    def test_passes_frequency(self) -> None:
        """Test frequency is correctly passed."""
        repo = get_repository(
            data_source=DataSource.LOCAL,
            frequency=Frequency.WEEKLY,
            file_path=Path("/fake/path.pkl"),
        )

        assert repo.frequency == Frequency.WEEKLY
