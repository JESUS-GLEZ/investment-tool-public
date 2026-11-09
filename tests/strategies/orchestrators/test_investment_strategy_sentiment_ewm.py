"""Unit tests for InvestmentStrategySentimentEWM."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.strategies.orchestrators.investment_strategy_sentiment_ewm import (
    InvestmentStrategySentimentEWM,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.sentiment_ewm_prediction_strategy import (
    SentimentEWMPredictionStrategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_instance() -> Instance:
    """Build a minimal Instance with a current trading period."""
    inst = MagicMock(spec=Instance)
    period = MagicMock()
    period.trading_date = pd.Timestamp("2024-07-01")
    period.decision_time = None
    period.eligible_symbols = None
    period.minimum_return_periods = None
    inst.current_trading_period = period
    inst.current_trading_period.execution_eligible_symbols = None
    return inst


def _make_stock(symbol: str, n_periods: int = 24) -> Stock:
    """Create a Stock with *n* months of synthetic data."""
    dates = pd.date_range("2022-01-01", periods=n_periods, freq="MS")
    rng = np.random.default_rng(42)
    close = 100.0 * np.cumprod(1 + rng.normal(0.01, 0.03, n_periods))
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


# ── SentimentEWMPredictionStrategy Tests ─────────────────────────────────


class TestSentimentEWMPredictionStrategy:
    """Test the regression-calibrated prediction strategy directly."""

    def test_predict_returns_no_sentiment_falls_back_to_ewm(self) -> None:
        """When no sentiment scores exist, prediction should approximate EWM."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores={"AAPL": {}},
            ewm_span=3,
            fallback_tilt=0.0,
        )
        preds = strategy.predict_returns()
        assert "AAPL" in preds
        assert isinstance(preds["AAPL"], float)
        assert np.isfinite(preds["AAPL"])

    def test_predict_returns_with_sentiment_produces_finite(self) -> None:
        """With sentiment, predictions should still be finite numbers."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        rng = np.random.default_rng(123)
        sentiments = {d.date(): float(rng.uniform(-1, 1)) for d in dates}

        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores={"AAPL": sentiments},
            ewm_span=3,
        )
        preds = strategy.predict_returns()
        assert "AAPL" in preds
        assert np.isfinite(preds["AAPL"])

    def test_calibration_diagnostics_populated(self) -> None:
        """After prediction, calibration_diagnostics should be populated."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        rng = np.random.default_rng(99)
        sentiments = {d.date(): float(rng.uniform(-1, 1)) for d in dates}

        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores={"AAPL": sentiments},
            ewm_span=3,
        )
        strategy.predict_returns()

        assert "AAPL" in strategy.calibration_diagnostics
        diag = strategy.calibration_diagnostics["AAPL"]
        assert "alpha" in diag
        assert "beta_ewm" in diag
        assert "beta_interaction" in diag
        assert "r_squared" in diag
        assert "calibrated" in diag

    def test_insufficient_data_uses_fallback(self) -> None:
        """With fewer than 12 periods, should use fallback (not regression)."""
        stocks = [_make_stock("AAPL", n_periods=5)]
        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores={"AAPL": {}},
            ewm_span=3,
            fallback_tilt=0.1,
        )
        preds = strategy.predict_returns()
        diag = strategy.calibration_diagnostics["AAPL"]
        assert diag["calibrated"] == 0.0
        assert np.isnan(diag["r_squared"])
        assert np.isfinite(preds["AAPL"])

    def test_calibrated_when_enough_data(self) -> None:
        """With >= 12 periods, the regression should be calibrated."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        dates = pd.date_range("2022-01-01", periods=24, freq="MS")
        rng = np.random.default_rng(42)
        sentiments = {d.date(): float(rng.uniform(-0.5, 0.5)) for d in dates}

        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores={"AAPL": sentiments},
            ewm_span=3,
        )
        strategy.predict_returns()

        diag = strategy.calibration_diagnostics["AAPL"]
        assert diag["calibrated"] == 1.0
        assert np.isfinite(diag["r_squared"])
        assert "effective_tilt" in diag

    def test_different_spans_produce_different_predictions(self) -> None:
        """EWM span 3 vs 12 should give different predictions."""
        stocks = [_make_stock("AAPL", n_periods=24)]
        sentiments: dict[str, dict[date, float]] = {"AAPL": {}}

        s3 = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores=sentiments,
            ewm_span=3,
        )
        s12 = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores=sentiments,
            ewm_span=12,
        )
        p3 = s3.predict_returns()
        p12 = s12.predict_returns()
        assert p3["AAPL"] != p12["AAPL"]

    def test_empty_returns_gives_zero(self) -> None:
        """Stock with all-NaN returns should predict 0.0."""
        dates = pd.date_range("2024-01-01", periods=3, freq="MS")
        data = pd.DataFrame(
            {
                StockFields.OPEN: [np.nan] * 3,
                StockFields.HIGH: [np.nan] * 3,
                StockFields.LOW: [np.nan] * 3,
                StockFields.CLOSE: [np.nan] * 3,
                StockFields.VOLUME: [0] * 3,
            },
            index=dates,
        )
        stock = object.__new__(Stock)
        stock.symbol = "EMPTY"
        stock.data = data
        stock.data[StockFields.RETURNS] = np.nan

        strategy = SentimentEWMPredictionStrategy(
            stocks=[stock],
            sentiment_scores={"EMPTY": {}},
            ewm_span=3,
        )
        preds = strategy.predict_returns()
        assert preds["EMPTY"] == 0.0

    def test_sentiment_series_clamps_to_range(self) -> None:
        """Sentiment values outside [-1, 1] should be clamped."""
        idx = pd.to_datetime(["2024-01-01", "2024-02-01"])
        scores = {date(2024, 1, 1): 5.0, date(2024, 2, 1): -3.0}
        result = SentimentEWMPredictionStrategy._build_sentiment_series(scores, idx)
        assert result.iloc[0] == 1.0
        assert result.iloc[1] == -1.0

    def test_multi_asset_prediction(self) -> None:
        """Strategy should handle multiple assets independently."""
        stocks = [_make_stock("AAPL", 24), _make_stock("GOOG", 24)]
        sentiments = {"AAPL": {}, "GOOG": {}}

        strategy = SentimentEWMPredictionStrategy(
            stocks=stocks,
            sentiment_scores=sentiments,
            ewm_span=3,
        )
        preds = strategy.predict_returns()
        assert "AAPL" in preds
        assert "GOOG" in preds
        assert np.isfinite(preds["AAPL"])
        assert np.isfinite(preds["GOOG"])


# ── Investment Strategy Contract Tests ───────────────────────────────────


class TestSentimentEWMContract:
    """Test the investment strategy initialisation and configuration."""

    def test_init_defaults(self) -> None:
        inst = _make_instance()
        stocks = [_make_stock("AAPL")]
        strategy = InvestmentStrategySentimentEWM(instance=inst, stocks=stocks, news={})
        assert strategy.ewm_span == 6
        assert strategy.fallback_tilt == 0.0
        assert strategy.population_size == 300
        assert strategy.n_generations == 6_000

    def test_custom_parameters(self) -> None:
        inst = _make_instance()
        stocks = [_make_stock("AAPL")]
        strategy = InvestmentStrategySentimentEWM(
            instance=inst,
            stocks=stocks,
            news={},
            ewm_span=6,
            fallback_tilt=0.1,
            population_size=200,
            n_generations=10_000,
        )
        assert strategy.ewm_span == 6
        assert strategy.fallback_tilt == 0.1
        assert strategy.population_size == 200
        assert strategy.n_generations == 10_000

    def test_predict_returns_delegates_to_sentiment_ewm_strategy(self) -> None:
        """The _predict_returns method should use SentimentEWMPredictionStrategy."""
        inst = _make_instance()
        stocks = [_make_stock("AAPL", n_periods=24)]
        strategy = InvestmentStrategySentimentEWM(instance=inst, stocks=stocks, news={})

        mock_preds = {"AAPL": 0.05}
        mock_diag = {"AAPL": {"calibrated": 1.0, "r_squared": 0.1}}

        with patch(
            "investment_tool.strategies.orchestrators"
            ".investment_strategy_sentiment_ewm.SentimentEWMPredictionStrategy"
        ) as mock_cls:
            mock_instance = MagicMock()
            mock_instance.predict_returns.return_value = mock_preds
            mock_instance.calibration_diagnostics = mock_diag
            mock_cls.return_value = mock_instance

            result = strategy._predict_returns({"AAPL": {}})

        assert result == mock_preds
        assert strategy._calibration_diagnostics == mock_diag
