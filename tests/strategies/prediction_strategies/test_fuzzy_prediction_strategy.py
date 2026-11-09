"""Formula reconstruction, causal training alignment and ablation controls."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.fuzzy_prediction_strategy import (
    FuzzyPredictionStrategy,
    FuzzySettings,
    PooledRidgePredictionStrategy,
    ReturnBaselinePredictionStrategy,
    contiguous_returns,
    fuzzy_features,
    fuzzy_forecast,
    trend_correction,
)


def stock(symbol="TEST", returns=None):
    if returns is None:
        returns = np.random.default_rng(47).normal(0.01, 0.05, 30)
    close = 100 * np.cumprod(np.r_[1.0, 1 + np.asarray(returns)])
    data = pd.DataFrame(
        {"close": close, "open": close, "high": close, "low": close, "volume": 1000},
        index=pd.date_range("2020-01-01", periods=len(close), freq="MS"),
    )
    return Stock(symbol, data)


def test_equation_410_uses_each_lag_not_experiment_index():
    assert trend_correction(np.array([1.0, 3.0, -2.0, 4.0])) == pytest.approx(-14 / 9)
    assert trend_correction(np.array([2.0] * 6)) == pytest.approx(0)


@pytest.mark.parametrize("settings", [FuzzySettings(), FuzzySettings(beta=0)])
def test_centroid_matches_independent_quadrature(settings):
    history = np.array([-18.0, -4.0, 2.0, 7.0, 24.0, 5.0, -9.0])
    low, high = min(history) - 24**2 / 100, max(history) + 24**2 / 100
    centers = np.linspace(low, high, 10)
    centers = (centers[:-1] + centers[1:]) / 2
    assigned = [int(np.argmin(abs(centers - value))) for value in history]
    widths = np.array(
        [
            settings.sigma + settings.beta * assigned.count(i) / len(history)
            for i in range(9)
        ]
    )

    def aggregate(x):
        return max(
            np.exp(-((x - centers[i]) ** 2) / (2 * widths[i] ** 2)) / widths[i]
            for i in assigned[-3:]
        )

    centroid = quad(lambda x: x * aggregate(x), low, high, epsabs=1e-9)[0]
    centroid /= quad(aggregate, low, high, epsabs=1e-9)[0]
    expected = (centroid + trend_correction(history)) / 100
    assert fuzzy_forecast(history / 100, settings) == pytest.approx(expected, abs=2e-6)


@pytest.mark.parametrize("value", [0.0, 0.02, -0.04])
def test_constant_history_is_finite_in_decimal_units(value):
    settings = FuzzySettings()
    result = fuzzy_forecast(np.full(20, value), settings)
    assert np.isfinite(result)
    # With nonzero margins, the centroid need not equal the constant return.
    assert abs(result - value) < 0.001
    features = fuzzy_features(np.full(20, value), settings)
    assert np.isfinite(features).all()
    assert features[-2] == pytest.approx(value)
    assert features[-1] == pytest.approx(0)


def test_missing_period_restarts_lag_history_without_bridging_gap():
    recent = np.array([0.02, 0.03, -0.01, 0.04])
    with_gap = np.r_[9.0, -0.9, np.nan, recent]
    np.testing.assert_array_equal(contiguous_returns(with_gap), recent)
    assert fuzzy_forecast(with_gap, FuzzySettings()) == fuzzy_forecast(
        recent, FuzzySettings()
    )
    np.testing.assert_array_equal(
        fuzzy_features(with_gap, FuzzySettings()),
        fuzzy_features(recent, FuzzySettings()),
    )
    assert np.isnan(fuzzy_forecast(np.array([0.01, np.nan]), FuzzySettings()))


def test_trend_ablation_changes_only_equation_410():
    values = np.array([-0.01, 0.05, -0.03, 0.08, 0.02])
    settings = FuzzySettings()
    difference = fuzzy_forecast(values, settings) - fuzzy_forecast(
        values, replace(settings, trend=False)
    )
    assert difference == pytest.approx(trend_correction(values * 100) / 100)


def test_fixed_width_is_beta_zero_and_has_normalised_ordered_features():
    values = np.array([-0.03, 0.01, 0.08, 0.06, -0.02])
    features = fuzzy_features(values, FuzzySettings(beta=0))
    np.testing.assert_allclose(features[:-2].reshape(3, 9).sum(axis=1), 1)
    reversed_features = fuzzy_features(values[::-1], FuzzySettings(beta=0))
    assert not np.allclose(features[:-2], reversed_features[:-2])
    assert not np.allclose(features, fuzzy_features(values, FuzzySettings()))


def test_baselines_and_strategies_return_decimal_returns():
    asset = stock(returns=np.full(20, 0.02))
    assert ReturnBaselinePredictionStrategy([asset], zero=True).predict_returns() == {
        "TEST": 0
    }
    assert ReturnBaselinePredictionStrategy([asset]).predict_returns()[
        "TEST"
    ] == pytest.approx(0.02)
    assert FuzzyPredictionStrategy([asset]).predict_returns()["TEST"] == pytest.approx(
        0.02, abs=0.001
    )


class RecordingModel:
    def fit(self, features, targets):
        self.features = features.copy()
        self.targets = targets.copy()

    def predict(self, features):
        self.current = features.copy()
        return np.zeros(len(features))


@pytest.mark.parametrize("fuzzy", [False, True])
def test_training_features_stop_before_next_return_label(fuzzy):
    returns = np.arange(1, 11) / 100
    first, changed = RecordingModel(), RecordingModel()
    with patch(
        "investment_tool.strategies.prediction_strategies.pooled_return_regression.make_pipeline",
        return_value=first,
    ):
        PooledRidgePredictionStrategy(
            [stock(returns=returns)], fuzzy=fuzzy, min_train_history=3
        ).predict_returns()
    returns[-1] = -0.5
    with patch(
        "investment_tool.strategies.prediction_strategies.pooled_return_regression.make_pipeline",
        return_value=changed,
    ):
        PooledRidgePredictionStrategy(
            [stock(returns=returns)], fuzzy=fuzzy, min_train_history=3
        ).predict_returns()
    # Even the final training feature must not see its own target.
    np.testing.assert_allclose(first.features, changed.features)
    np.testing.assert_allclose(first.targets, np.arange(4, 11) / 100)
    assert changed.targets[-1] == pytest.approx(-0.5)
    assert not np.allclose(first.current, changed.current)
    if not fuzzy:
        np.testing.assert_allclose(first.features[0, :3], [0.01, 0.02, 0.03])
        np.testing.assert_allclose(first.current[0, :3], [0.08, 0.09, 0.10])


@pytest.mark.parametrize("fuzzy", [False, True])
def test_pooled_training_is_deterministic_and_stock_order_independent(fuzzy):
    assets = [stock("A"), stock("B", np.random.default_rng(9).normal(0, 0.1, 25))]
    forward = PooledRidgePredictionStrategy(assets, fuzzy=fuzzy).predict_returns()
    backward = PooledRidgePredictionStrategy(
        assets[::-1], fuzzy=fuzzy
    ).predict_returns()
    assert forward == pytest.approx(backward)
    assert all(np.isfinite(list(forward.values())))


def test_short_history_fallback_is_explicit_historical_mean():
    asset = stock(returns=[0.01, 0.02, 0.03])
    assert PooledRidgePredictionStrategy([asset]).predict_returns()[
        "TEST"
    ] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "kwargs",
    [{"sigma": 0}, {"beta": -1}, {"partitions": 1}, {"lags": 0}, {"sigma": np.nan}],
)
def test_invalid_settings_fail_early(kwargs):
    with pytest.raises(ValueError):
        FuzzySettings(**kwargs)


def test_pooled_training_does_not_bridge_a_missing_return():
    asset = stock(returns=np.arange(1, 13) / 100)
    asset.data["returns"] = np.r_[
        np.nan, np.arange(1, 6) / 100, np.nan, np.arange(7, 13) / 100
    ]
    model = RecordingModel()
    with patch(
        "investment_tool.strategies.prediction_strategies.pooled_return_regression.make_pipeline",
        return_value=model,
    ):
        PooledRidgePredictionStrategy(
            [asset], fuzzy=False, min_train_history=3
        ).predict_returns()
    np.testing.assert_allclose(model.targets, [0.04, 0.05, 0.10, 0.11, 0.12])
    np.testing.assert_allclose(np.diff(model.features[:, :3], axis=1), 0.01)


def test_centroid_remains_resolved_for_extremely_wide_historical_domain():
    # A 1000% return creates a domain much wider than sigma=10 percentage points.
    values = np.array([10.0, -0.2, 0.03, -0.02, 0.01])
    settings = FuzzySettings(trend=False)
    fine = replace(settings, grid_points=20001)
    assert fuzzy_forecast(values, settings) == pytest.approx(
        fuzzy_forecast(values, fine), abs=2e-5
    )
