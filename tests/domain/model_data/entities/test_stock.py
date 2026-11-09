"""Unit tests for Stock entity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import StockFields
from investment_tool.domain.model_data.entities.stock import Stock


def _create_sample_ohlcv(n_days: int = 30) -> pd.DataFrame:
    """Create sample OHLCV data for testing.

    Parameters
    ----------
    n_days : int
        Number of days of data.

    Returns
    -------
    pd.DataFrame
        OHLCV data with DatetimeIndex.
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    np.random.seed(42)

    # Generate realistic-ish price data
    base_price = 100.0
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    prices = base_price * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            StockFields.OPEN: prices * (1 + np.random.randn(n_days) * 0.001),
            StockFields.HIGH: prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            StockFields.LOW: prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            StockFields.CLOSE: prices,
            StockFields.VOLUME: np.random.randint(1000000, 10000000, n_days),
        },
        index=dates,
    )

    return data


class TestStockInitialization:
    """Tests for Stock initialization and data preparation."""

    def test_basic_initialization(self) -> None:
        """Test basic stock creation."""
        data = _create_sample_ohlcv()
        stock = Stock("AAPL", data)

        assert stock.symbol == "AAPL"
        assert len(stock.data) == 30
        assert isinstance(stock.data.index, pd.DatetimeIndex)

    def test_str_representation(self) -> None:
        """Test __str__ returns symbol."""
        data = _create_sample_ohlcv()
        stock = Stock("MSFT", data)

        assert str(stock) == "MSFT"

    def test_repr_representation(self) -> None:
        """Test __repr__ includes symbol and period count."""
        data = _create_sample_ohlcv(20)
        stock = Stock("GOOG", data)

        repr_str = repr(stock)
        assert "GOOG" in repr_str
        assert "20" in repr_str

    def test_accepts_date_column(self) -> None:
        """Test initialization with 'date' column instead of index."""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        data = pd.DataFrame(
            {
                StockFields.DATE: dates,
                StockFields.OPEN: np.random.rand(10) * 100,
                StockFields.HIGH: np.random.rand(10) * 100,
                StockFields.LOW: np.random.rand(10) * 100,
                StockFields.CLOSE: np.random.rand(10) * 100,
                StockFields.VOLUME: np.random.randint(1000, 10000, 10),
            }
        )

        stock = Stock("TEST", data)

        assert isinstance(stock.data.index, pd.DatetimeIndex)
        assert StockFields.DATE not in stock.data.columns

    def test_raises_without_datetime_index(self) -> None:
        """Test error raised when no DatetimeIndex or date column."""
        data = pd.DataFrame(
            {
                StockFields.OPEN: [100, 101, 102],
                StockFields.CLOSE: [101, 102, 103],
            }
        )

        with pytest.raises(ValueError, match="DatetimeIndex"):
            Stock("BAD", data)

    def test_data_is_sorted_chronologically(self) -> None:
        """Test that data is sorted by date ascending."""
        # Create with reversed order
        data = _create_sample_ohlcv(10)
        data = data.iloc[::-1]  # Reverse

        stock = Stock("SORT", data)

        # Should be sorted ascending
        assert stock.data.index.is_monotonic_increasing

    def test_data_copy_is_made(self) -> None:
        """Test that original data is not modified."""
        original_data = _create_sample_ohlcv(10)
        original_copy = original_data.copy()

        # Original should be unchanged
        pd.testing.assert_frame_equal(original_data, original_copy)


class TestStockReturns:
    """Tests for return calculations."""

    @pytest.mark.parametrize("bad_volume", [0, -1, np.nan, np.inf])
    def test_no_trade_bar_cannot_create_prices_or_gap_returns(self, bad_volume):
        data = _create_sample_ohlcv(5)
        data["volume"] = data["volume"].astype(float)
        data.iloc[2, data.columns.get_loc("volume")] = bad_volume
        original = data.copy()
        stock = Stock("GAP", data)
        assert stock.data.iloc[2][["open", "high", "low", "close"]].isna().all()
        assert stock.data["returns"].iloc[2:4].isna().all()
        assert stock.data["log_returns"].iloc[2:4].isna().all()
        assert stock.data["returns"].iloc[4] == pytest.approx(
            data.close.iloc[4] / data.close.iloc[3] - 1
        )
        pd.testing.assert_frame_equal(data, original)

    def test_missing_close_is_not_implicitly_forward_filled(self):
        data = _create_sample_ohlcv(5)
        data.loc[data.index[2], "close"] = np.nan
        stock = Stock("GAP", data)
        assert stock.data["returns"].iloc[2:4].isna().all()

    def test_simple_returns_calculated(self) -> None:
        """Test that simple returns are calculated correctly."""
        data = _create_sample_ohlcv()
        stock = Stock("AAPL", data)

        assert StockFields.RETURNS in stock.data.columns

        # First return should be NaN
        assert pd.isna(stock.data[StockFields.RETURNS].iloc[0])

        # Manual verification of second return
        expected_return = (
            stock.data[StockFields.CLOSE].iloc[1]
            / stock.data[StockFields.CLOSE].iloc[0]
            - 1
        )
        actual_return = stock.data[StockFields.RETURNS].iloc[1]
        np.testing.assert_almost_equal(actual_return, expected_return)

    def test_log_returns_calculated(self) -> None:
        """Test that log returns are calculated correctly."""
        data = _create_sample_ohlcv()
        stock = Stock("AAPL", data)

        assert StockFields.LOG_RETURNS in stock.data.columns

        # Manual verification
        expected_log_return = np.log(
            stock.data[StockFields.CLOSE].iloc[1]
            / stock.data[StockFields.CLOSE].iloc[0]
        )
        actual_log_return = stock.data[StockFields.LOG_RETURNS].iloc[1]
        np.testing.assert_almost_equal(actual_log_return, expected_log_return)

    def test_returns_sum_approximately_to_log_returns(self) -> None:
        """Test relationship between simple and log returns."""
        data = _create_sample_ohlcv()
        stock = Stock("AAPL", data)

        # For small returns, log(1+r) ≈ r
        simple_returns = stock.data[StockFields.RETURNS].dropna()
        log_returns = stock.data[StockFields.LOG_RETURNS].dropna()

        # Check relationship: log_return = log(1 + simple_return)
        expected = np.log(1 + simple_returns)
        np.testing.assert_array_almost_equal(log_returns.values, expected.values)


class TestStockMovingAverages:
    """Tests for moving average calculations."""

    def test_default_moving_averages_exist(self) -> None:
        """Test that default MA windows are calculated."""
        data = _create_sample_ohlcv(250)  # Need enough data for MA_200
        stock = Stock("AAPL", data)

        expected_mas = ["ma_5", "ma_10", "ma_20", "ma_50", "ma_100", "ma_200"]
        for ma in expected_mas:
            assert ma in stock.data.columns

    def test_moving_average_values(self) -> None:
        """Test moving average calculation accuracy."""
        data = _create_sample_ohlcv(30)
        stock = Stock("AAPL", data)

        # Verify MA_5 calculation manually
        close_prices = stock.data[StockFields.CLOSE]
        expected_ma5 = close_prices.rolling(5).mean()
        actual_ma5 = stock.data["ma_5"]

        pd.testing.assert_series_equal(actual_ma5, expected_ma5, check_names=False)

    def test_ma_nan_for_insufficient_data(self) -> None:
        """Test that MA is NaN when insufficient data points."""
        data = _create_sample_ohlcv(30)
        stock = Stock("AAPL", data)

        # MA_10 should have NaN for first 9 values
        assert stock.data["ma_10"].iloc[:9].isna().all()
        assert not stock.data["ma_10"].iloc[9:].isna().any()


class TestStockVolatility:
    """Tests for volatility calculation."""

    def test_volatility_calculated(self) -> None:
        """Test that volatility column exists."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        assert StockFields.VOLATILITY in stock.data.columns

    def test_volatility_is_positive(self) -> None:
        """Test that volatility values are non-negative."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        vol = stock.data[StockFields.VOLATILITY].dropna()
        assert (vol >= 0).all()

    def test_volatility_formula(self) -> None:
        """Test volatility calculation formula."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        # Volatility = rolling std * sqrt(window)
        returns = stock.data[StockFields.RETURNS]
        window = 20
        expected_vol = returns.rolling(window).std() * np.sqrt(window)
        actual_vol = stock.data[StockFields.VOLATILITY]

        pd.testing.assert_series_equal(actual_vol, expected_vol, check_names=False)


class TestStockRSI:
    """Tests for RSI calculation."""

    def test_rsi_calculated(self) -> None:
        """Test that RSI column exists."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        assert StockFields.RSI in stock.data.columns

    def test_rsi_bounded(self) -> None:
        """Test that RSI is between 0 and 100."""
        data = _create_sample_ohlcv(100)
        stock = Stock("AAPL", data)

        rsi = stock.data[StockFields.RSI].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_rsi_with_all_gains(self) -> None:
        """Test RSI approaches 100 with only gains."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = np.linspace(100, 130, 30)  # Steadily increasing
        data = pd.DataFrame(
            {
                StockFields.OPEN: prices,
                StockFields.HIGH: prices * 1.01,
                StockFields.LOW: prices * 0.99,
                StockFields.CLOSE: prices,
                StockFields.VOLUME: [1000000] * 30,
            },
            index=dates,
        )

        stock = Stock("UP", data)

        # RSI should be very high (close to 100)
        rsi_late = stock.data[StockFields.RSI].iloc[-1]
        assert rsi_late > 90

    def test_rsi_with_all_losses(self) -> None:
        """Test RSI approaches 0 with only losses."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = np.linspace(130, 100, 30)  # Steadily decreasing
        data = pd.DataFrame(
            {
                StockFields.OPEN: prices,
                StockFields.HIGH: prices * 1.01,
                StockFields.LOW: prices * 0.99,
                StockFields.CLOSE: prices,
                StockFields.VOLUME: [1000000] * 30,
            },
            index=dates,
        )

        stock = Stock("DOWN", data)

        # RSI should be very low (close to 0)
        rsi_late = stock.data[StockFields.RSI].iloc[-1]
        assert rsi_late < 10


class TestStockBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands_calculation(self) -> None:
        """Test Bollinger Bands can be calculated."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        # Call the method explicitly
        stock.calculate_bollinger_bands()

        assert StockFields.MIDDLE_BAND in stock.data.columns
        assert StockFields.UPPER_BAND in stock.data.columns
        assert StockFields.LOWER_BAND in stock.data.columns

    def test_band_ordering(self) -> None:
        """Test that upper > middle > lower."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)
        stock.calculate_bollinger_bands()

        # Get non-NaN values
        middle = stock.data[StockFields.MIDDLE_BAND].dropna()
        upper = stock.data[StockFields.UPPER_BAND].dropna()
        lower = stock.data[StockFields.LOWER_BAND].dropna()

        assert (upper >= middle).all()
        assert (middle >= lower).all()

    def test_custom_parameters(self) -> None:
        """Test Bollinger Bands with custom parameters."""
        data = _create_sample_ohlcv(50)
        stock = Stock("AAPL", data)

        stock.calculate_bollinger_bands(window=10, num_std_dev=3)

        # Middle band should be 10-period MA
        close_prices = stock.data[StockFields.CLOSE]
        expected_middle = close_prices.rolling(10).mean()

        pd.testing.assert_series_equal(
            stock.data[StockFields.MIDDLE_BAND], expected_middle, check_names=False
        )
