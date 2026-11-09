"""Unit tests for prediction strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)


def _create_stock(symbol: str, returns: list[float]) -> Stock:
    """Create a Stock with specified returns for testing.

    Parameters
    ----------
    symbol : str
        Stock symbol.
    returns : list[float]
        List of return values.

    Returns
    -------
    Stock
        Stock object with computed data.
    """
    n = len(returns) + 1  # Need one extra for pct_change
    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    # Create prices from returns
    prices = [100.0]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    data = pd.DataFrame(
        {
            StockFields.OPEN: prices,
            StockFields.HIGH: [p * 1.01 for p in prices],
            StockFields.LOW: [p * 0.99 for p in prices],
            StockFields.CLOSE: prices,
            StockFields.VOLUME: [1000000] * n,
        },
        index=dates,
    )

    return Stock(symbol, data)


class TestAveragePredictionStrategy:
    """Tests for AveragePredictionStrategy (exponentially-weighted mean)."""

    def test_basic_prediction(self) -> None:
        """Test that prediction is positive for positive returns."""
        stock = _create_stock("AAPL", [0.01, 0.02, 0.03])
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        assert "AAPL" in predictions
        # EWM gives higher weight to recent obs, so result > flat mean of 0.02
        assert predictions["AAPL"] > 0.0
        assert predictions["AAPL"] < 0.05  # sanity bound

    def test_multiple_stocks(self) -> None:
        """Test predictions for multiple stocks."""
        stock1 = _create_stock("AAPL", [0.01, 0.02, 0.03])
        stock2 = _create_stock("MSFT", [0.05, 0.05, 0.05])

        strategy = AveragePredictionStrategy([stock1, stock2])
        predictions = strategy.predict_returns()

        assert len(predictions) == 2
        assert "AAPL" in predictions
        assert "MSFT" in predictions
        # Constant returns → EWM mean ≈ flat mean
        assert abs(predictions["MSFT"] - 0.05) < 0.01

    def test_negative_returns(self) -> None:
        """Test with negative returns."""
        stock = _create_stock("DOWN", [-0.01, -0.02, -0.03])
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        assert predictions["DOWN"] < 0

    def test_mixed_returns(self) -> None:
        """Test with mixed positive and negative returns."""
        stock = _create_stock("MIX", [0.05, -0.05, 0.03, -0.03])
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        # EWM of mixed returns — should be close to zero
        assert abs(predictions["MIX"]) < 0.05

    def test_empty_returns(self) -> None:
        """Test behavior with very short history."""
        # Stock with only one price point (no returns)
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        data = pd.DataFrame(
            {
                StockFields.OPEN: [100.0],
                StockFields.HIGH: [101.0],
                StockFields.LOW: [99.0],
                StockFields.CLOSE: [100.0],
                StockFields.VOLUME: [1000000],
            },
            index=dates,
        )

        stock = Stock("SHORT", data)
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        # Should return 0.0 for empty returns
        assert predictions["SHORT"] == 0.0

    def test_returns_are_float(self) -> None:
        """Test that predictions are floats."""
        stock = _create_stock("TEST", [0.01, 0.02])
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        assert isinstance(predictions["TEST"], float)

    def test_returns_dict_structure(self) -> None:
        """Test return value is dict with string keys."""
        stock = _create_stock("AAPL", [0.01, 0.02])
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        assert isinstance(predictions, dict)
        for key in predictions:
            assert isinstance(key, str)

    def test_stocks_stored(self) -> None:
        """Test that stocks are stored during init."""
        stock = _create_stock("AAPL", [0.01, 0.02, 0.03])
        strategy = AveragePredictionStrategy([stock])

        assert len(strategy.stocks) == 1
        assert strategy.stocks[0].symbol == "AAPL"

    def test_nan_handling(self) -> None:
        """Test that NaN returns are dropped from calculation."""
        stock = _create_stock("NAN", [0.01, 0.02, 0.03])
        # First return is always NaN due to pct_change
        strategy = AveragePredictionStrategy([stock])

        predictions = strategy.predict_returns()

        # Should still work (NaN dropped via dropna)
        assert not np.isnan(predictions["NAN"])
