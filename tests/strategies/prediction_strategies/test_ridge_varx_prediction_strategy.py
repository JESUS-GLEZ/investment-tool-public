"""Unit tests for RidgeVARXPredictionStrategy.

Tests the multi-output Ridge VARX prediction pipeline: data preparation,
lagged feature construction, and the predict_returns contract.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from investment_tool.constants import StockFields
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.ridge_varx_prediction_strategy import (
    RidgeVARXPredictionStrategy,
)

_MODULE = (
    "investment_tool.strategies.prediction_strategies.ridge_varx_prediction_strategy"
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_stock(
    symbol: str,
    n_periods: int = 24,
    trend: float = 0.01,
) -> Stock:
    """Create a Stock with *n_periods* months of trending close prices.

    The Stock constructor auto-computes returns, moving averages, etc.
    """
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    close = 100.0 * np.cumprod(1 + np.full(n_periods, trend))
    data = pd.DataFrame(
        {
            StockFields.OPEN: close * 0.99,
            StockFields.HIGH: close * 1.02,
            StockFields.LOW: close * 0.98,
            StockFields.CLOSE: close,
            StockFields.VOLUME: np.full(n_periods, 1_000_000),
        },
        index=dates,
    )
    return Stock(symbol, data)


def _make_sentiment(
    symbols: list[str],
    n_dates: int = 12,
    score: float = 0.5,
) -> dict[str, dict[date, float]]:
    """Create uniform sentiment scores for testing."""
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="MS")
    return {symbol: {d.date(): score for d in dates} for symbol in symbols}


# ── Basic Contract Tests ─────────────────────────────────────────────────────


class TestRidgeVARXContract:
    """Test the predict_returns contract: returns dict[str, float]."""

    def test_returns_dict_of_floats(self) -> None:
        symbols = ["AAPL", "GOOG"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        result = strategy.predict_returns()

        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, float)

    def test_returns_one_prediction_per_stock(self) -> None:
        symbols = ["AAPL", "GOOG", "MSFT"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        result = strategy.predict_returns()

        assert set(result.keys()) == set(symbols)

    def test_single_stock(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiment = _make_sentiment(["AAPL"])

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        result = strategy.predict_returns()

        assert "AAPL" in result
        assert isinstance(result["AAPL"], float)

    def test_predictions_are_finite(self) -> None:
        symbols = ["AAPL", "GOOG"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        result = strategy.predict_returns()

        for val in result.values():
            assert np.isfinite(val), f"Non-finite prediction: {val}"


# ── Data Preparation Tests ───────────────────────────────────────────────────


class TestRidgeVARXDataPreparation:
    """Test the _prepare_data method."""

    def test_prepare_data_returns_two_dataframes(self) -> None:
        symbols = ["AAPL", "GOOG"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        returns_df, exog_df = strategy._prepare_data()

        assert isinstance(returns_df, pd.DataFrame)
        assert isinstance(exog_df, pd.DataFrame)

    def test_returns_df_columns_match_symbols(self) -> None:
        symbols = ["AAPL", "GOOG"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        returns_df, _ = strategy._prepare_data()

        assert list(returns_df.columns) == symbols

    def test_recent_listing_does_not_erase_training_history(self) -> None:
        established = _make_stock("AAPL", n_periods=24, trend=0.02)
        recent = _make_stock("NEW", n_periods=24).data.iloc[-3:]
        newcomer = Stock("NEW", recent)
        strategy = RidgeVARXPredictionStrategy([established, newcomer], {})

        returns, _ = strategy._prepare_data()
        predictions = strategy.predict_returns()

        assert list(returns.columns) == ["AAPL"]
        assert len(returns) >= 12
        assert set(predictions) == {"AAPL"}
        assert predictions["AAPL"] != 0.0

    def test_exog_df_has_ma_and_sentiment_columns(self) -> None:
        symbols = ["AAPL"]
        stocks = [_make_stock(s, n_periods=24) for s in symbols]
        sentiment = _make_sentiment(symbols)

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        _, exog_df = strategy._prepare_data()

        expected_cols = {
            "AAPL_price_ma_ratio",
            f"AAPL_{StockFields.SENTIMENT_SCORE}",
        }
        assert expected_cols == set(exog_df.columns)

    def test_no_nan_in_returns_after_preparation(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiment = _make_sentiment(["AAPL"])

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        returns_df, _ = strategy._prepare_data()

        assert not returns_df.isna().any().any()

    def test_no_future_data_in_returns(self) -> None:
        """Forward-fill only — no backward-fill that would leak future data."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiment = _make_sentiment(["AAPL"])

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        returns_df, _ = strategy._prepare_data()

        # Returns are forward-filled then dropna → no NaN and chronological
        assert returns_df.index.is_monotonic_increasing


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestRidgeVARXEdgeCases:
    """Test edge cases and robustness."""

    def test_insufficient_data_returns_zeros(self) -> None:
        """With very few data points, should return zero predictions."""
        dates = pd.date_range("2024-01-01", periods=3, freq="MS")
        data = pd.DataFrame(
            {
                StockFields.OPEN: [100.0, 101.0, 102.0],
                StockFields.HIGH: [102.0, 103.0, 104.0],
                StockFields.LOW: [98.0, 99.0, 100.0],
                StockFields.CLOSE: [100.0, 101.0, 102.0],
                StockFields.VOLUME: [1e6, 1e6, 1e6],
            },
            index=dates,
        )
        stock = Stock("SHORT", data)
        sentiment = _make_sentiment(["SHORT"], n_dates=3)

        strategy = RidgeVARXPredictionStrategy([stock], sentiment, lag_order=3)
        result = strategy.predict_returns()

        # With lag_order=3 and only 3 periods, after dropna there's
        # not enough data → should return 0.0
        assert result["SHORT"] == 0.0

    def test_missing_sentiment_defaults_to_zero(self) -> None:
        """Missing sentiment dates should be filled with 0."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        # Empty sentiment — no dates at all
        sentiment: dict[str, dict[date, float]] = {"AAPL": {}}

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment)
        result = strategy.predict_returns()

        # Should still produce a prediction (sentiment filled with 0)
        assert "AAPL" in result
        assert np.isfinite(result["AAPL"])

    def test_asset_without_news_uses_neutral_sentiment(self) -> None:
        """Crypto and other news-free assets remain in a mixed universe."""
        stocks = [_make_stock("AAPL"), _make_stock("BTC/USD")]
        strategy = RidgeVARXPredictionStrategy(stocks, _make_sentiment(["AAPL"]))

        _, features = strategy._prepare_data()
        assert features[f"BTC/USD_{StockFields.SENTIMENT_SCORE}"].eq(0.0).all()

        predictions = strategy.predict_returns()
        assert set(predictions) == {"AAPL", "BTC/USD"}
        assert all(np.isfinite(value) for value in predictions.values())

    def test_custom_lag_order(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiment = _make_sentiment(["AAPL"])

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment, lag_order=5)
        assert strategy.lag_order == 5

        result = strategy.predict_returns()
        assert "AAPL" in result

    def test_custom_alpha(self) -> None:
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiment = _make_sentiment(["AAPL"])

        strategy = RidgeVARXPredictionStrategy(stocks, sentiment, alpha=1.0)
        assert strategy.alpha == 1.0

        result = strategy.predict_returns()
        assert "AAPL" in result


# ── Multi-Output Equivalence ─────────────────────────────────────────────────


class TestMultiOutputEquivalence:
    """Verify the multi-output fit produces reasonable results.

    The multi-output Ridge fit (single model) is mathematically identical
    to N separate single-output fits.  We verify the predictions are
    consistent across stocks.
    """

    def test_different_trends_produce_different_predictions(self) -> None:
        """Stocks with different trends should yield different predictions."""
        stock_up = _make_stock("UP", n_periods=24, trend=0.03)
        stock_down = _make_stock("DOWN", n_periods=24, trend=-0.02)
        sentiment = _make_sentiment(["UP", "DOWN"])

        strategy = RidgeVARXPredictionStrategy([stock_up, stock_down], sentiment)
        result = strategy.predict_returns()

        # Upward trending stock should have higher predicted return
        assert result["UP"] > result["DOWN"]

    def test_identical_stocks_produce_similar_predictions(self) -> None:
        """Two stocks with the same trend should produce similar predictions."""
        stock_a = _make_stock("A", n_periods=24, trend=0.01)
        stock_b = _make_stock("B", n_periods=24, trend=0.01)
        sentiment = _make_sentiment(["A", "B"], score=0.5)

        strategy = RidgeVARXPredictionStrategy([stock_a, stock_b], sentiment)
        result = strategy.predict_returns()

        # Should be very close (not identical due to numerical noise in
        # computing returns from slightly different price series)
        assert abs(result["A"] - result["B"]) < 0.01


class TestNextPeriodForecast:
    """Training must pair completed features with the following period's return."""

    @pytest.mark.parametrize("lag_order", [1, 3, 5])
    def test_training_and_forecast_dates_are_aligned(self, lag_order: int) -> None:
        # Monthly labels are period starts; the final row is completed April.
        dates = pd.date_range("2023-09-01", periods=8, freq="MS")
        returns = pd.DataFrame(
            {
                "A": [0.01, -0.02, 0.03, 0.04, -0.05, 0.06, 0.07, -0.08],
                "B": [0.08, 0.07, -0.06, 0.05, 0.04, -0.03, 0.02, 0.01],
            },
            index=dates,
        )
        exog = pd.DataFrame(
            {
                "ratio": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 9.0],
                "sentiment": [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.9],
            },
            index=dates,
        )
        strategy = RidgeVARXPredictionStrategy([], {}, lag_order=lag_order)
        scaler = StandardScaler()
        model = Ridge(alpha=strategy.alpha)
        with (
            patch.object(strategy, "_prepare_data", return_value=(returns, exog)),
            patch(f"{_MODULE}.StandardScaler", return_value=scaler),
            patch.object(scaler, "fit_transform", wraps=scaler.fit_transform) as fit,
            patch.object(scaler, "transform", wraps=scaler.transform) as transform,
            patch(f"{_MODULE}.Ridge", return_value=model),
            patch.object(model, "fit", wraps=model.fit) as model_fit,
        ):
            predictions = strategy.predict_returns()

        # Each training origin uses only its own and earlier observations;
        # its target is the next row, including April's observed return.
        origins = range(lag_order - 1, 7)
        expected_train = np.array(
            [
                [value for lag in range(lag_order) for value in returns.iloc[i - lag]]
                + exog.iloc[i].tolist()
                for i in origins
            ]
        )
        expected_forecast = np.array(
            [
                [value for lag in range(lag_order) for value in returns.iloc[7 - lag]]
                + exog.iloc[7].tolist()
            ]
        )
        np.testing.assert_allclose(fit.call_args.args[0], expected_train)
        np.testing.assert_allclose(
            model_fit.call_args.args[1], returns.iloc[lag_order:]
        )
        np.testing.assert_allclose(
            transform.call_args_list[-1].args[0], expected_forecast
        )
        # April's feature row is reserved for May, including for scaling.
        np.testing.assert_allclose(scaler.mean_, expected_train.mean(axis=0))
        assert set(predictions) == {"A", "B"}
        assert all(np.isfinite(value) for value in predictions.values())

    def test_forecasts_next_return_from_latest_sentiment(self) -> None:
        dates = pd.date_range("2022-01-01", periods=18, freq="MS")
        sentiment = np.random.default_rng(7).uniform(-0.8, 0.8, len(dates))
        # Known next-month relationship: r(t+1) = 0.02 + 0.1 * sentiment(t).
        returns = pd.DataFrame(
            {"A": np.r_[0.03, 0.02 + 0.1 * sentiment[:-1]]}, index=dates
        )
        exog = pd.DataFrame({"sentiment": sentiment}, index=dates)
        strategy = RidgeVARXPredictionStrategy([], {}, lag_order=1, alpha=0.0)
        with patch.object(strategy, "_prepare_data", return_value=(returns, exog)):
            result = strategy.predict_returns()

        assert result["A"] == pytest.approx(0.02 + 0.1 * sentiment[-1])
        assert result["A"] != pytest.approx(returns.iloc[-1, 0])

    def test_missing_historical_features_do_not_skip_a_target_period(self) -> None:
        dates = pd.date_range("2024-01-01", periods=6, freq="MS")
        returns = pd.DataFrame({"A": np.arange(6) * 0.01}, index=dates)
        exog = pd.DataFrame({"ratio": [1.0, 1.1, np.nan, 1.3, 1.4, 1.5]}, index=dates)
        strategy = RidgeVARXPredictionStrategy([], {}, lag_order=1)
        model = Ridge(alpha=strategy.alpha)
        with (
            patch.object(strategy, "_prepare_data", return_value=(returns, exog)),
            patch(f"{_MODULE}.Ridge", return_value=model),
            patch.object(model, "fit", wraps=model.fit) as fit,
        ):
            strategy.predict_returns()

        # February still targets March, even though March's feature row is
        # incomplete. Dropping feature rows before shifting would target April.
        np.testing.assert_allclose(
            fit.call_args.args[1], [[0.01], [0.02], [0.04], [0.05]]
        )

    @pytest.mark.parametrize("n_periods", [2, 3])
    def test_no_completed_training_pair_returns_zero(self, n_periods: int) -> None:
        dates = pd.date_range("2024-01-01", periods=n_periods, freq="MS")
        returns = pd.DataFrame({"A": np.full(n_periods, 0.01)}, index=dates)
        exog = pd.DataFrame({"sentiment": np.zeros(n_periods)}, index=dates)
        strategy = RidgeVARXPredictionStrategy([], {}, lag_order=3)
        with (
            patch.object(strategy, "_prepare_data", return_value=(returns, exog)),
            patch(f"{_MODULE}.Ridge") as model,
        ):
            assert strategy.predict_returns() == {"A": 0.0}
        model.assert_not_called()

    def test_missing_latest_features_does_not_forecast_an_older_month(self) -> None:
        dates = pd.date_range("2023-09-01", periods=8, freq="MS")
        returns = pd.DataFrame({"A": np.arange(8) * 0.01}, index=dates)
        exog = pd.DataFrame({"sentiment": [0.5] * 7 + [np.nan]}, index=dates)
        strategy = RidgeVARXPredictionStrategy([], {}, lag_order=3)
        with (
            patch.object(strategy, "_prepare_data", return_value=(returns, exog)),
            patch(f"{_MODULE}.Ridge") as model,
        ):
            assert strategy.predict_returns() == {"A": 0.0}
        model.assert_not_called()
