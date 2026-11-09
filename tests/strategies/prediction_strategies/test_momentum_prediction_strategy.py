"""Fixed blend identities, monthly signals and pooled next-return alignment."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.fuzzy_prediction_strategy import (
    ReturnBaselinePredictionStrategy,
)
from investment_tool.strategies.prediction_strategies.momentum_prediction_strategy import (
    MeanBlendPredictionStrategy,
    CalibratedMomentumPredictionStrategy,
    _ewm_last,
    momentum_12_2,
)
from investment_tool.strategies.prediction_strategies.pooled_return_regression import (
    PooledReturnRegression,
)
from tests.strategies.prediction_strategies.test_fuzzy_prediction_strategy import stock


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_blend_reuses_exact_existing_mean_and_ewm(weight):
    assets = [stock("A"), stock("B", np.linspace(-0.1, 0.2, 30))]
    mean = ReturnBaselinePredictionStrategy(assets).predict_returns()
    ewm = AveragePredictionStrategy(assets, span=12).predict_returns()
    blend = MeanBlendPredictionStrategy(
        assets, momentum_weight=weight
    ).predict_returns()
    for symbol in mean:
        assert blend[symbol] == pytest.approx(
            weight * ewm[symbol] + (1 - weight) * mean[symbol]
        )


@pytest.mark.parametrize("weight", [-0.01, 1.01, np.nan, np.inf])
def test_invalid_blend_weight(weight):
    with pytest.raises(ValueError, match="between 0 and 1"):
        MeanBlendPredictionStrategy([], momentum_weight=weight)


@pytest.mark.parametrize("span", [6, 12])
@pytest.mark.parametrize("length", [12, 30, 120])
def test_ewm_feature_matches_pandas_adjusted_weights(span, length):
    values = np.random.default_rng(42).normal(0.01, 0.1, length)
    assert _ewm_last(values, span) == pytest.approx(
        pd.Series(values).ewm(span=span).mean().iloc[-1]
    )


def test_12_2_uses_eleven_returns_and_excludes_latest_and_older_months():
    values = np.linspace(-0.05, 0.1, 15)
    expected = np.prod(1 + values[3:14]) - 1
    assert momentum_12_2(values) == pytest.approx(expected)
    changed = values.copy()
    changed[:3] = 10
    changed[-1] = 20
    assert momentum_12_2(changed) == pytest.approx(expected)
    changed[-2] += 0.5
    assert momentum_12_2(changed) != pytest.approx(expected)


@pytest.mark.parametrize("values", [np.ones(11), np.r_[np.ones(11), np.nan]])
def test_12_2_requires_contiguous_months(values):
    with pytest.raises(ValueError, match="12 contiguous"):
        momentum_12_2(values)


def test_four_feature_model_is_in_decimal_units_and_single_signal_is_calibrated():
    values = np.linspace(-0.1, 0.2, 20)
    model = CalibratedMomentumPredictionStrategy([])
    np.testing.assert_allclose(
        model._features(values),
        [
            pd.Series(values).ewm(span=6).mean().iloc[-1],
            pd.Series(values).ewm(span=12).mean().iloc[-1],
            np.prod(1 + values[-12:-1]) - 1,
            np.std(values[-12:], ddof=0),
        ],
    )
    single = CalibratedMomentumPredictionStrategy([], signals="12-2")
    np.testing.assert_allclose(single._features(values), [momentum_12_2(values)])


def test_training_rows_are_prefix_to_next_return_in_chronological_asset_batches():
    # The spy makes row/label alignment observable independently of sklearn.
    a = stock("A", np.arange(1, 9) / 100)
    b = stock("B", np.arange(11, 19) / 100)

    class RecordingModel:
        def fit(self, features, targets):
            self.features, self.targets = features, targets

        def predict(self, features):
            self.current = features
            return np.zeros(len(features))

    model = RecordingModel()
    with patch(
        "investment_tool.strategies.prediction_strategies.pooled_return_regression.make_pipeline",
        return_value=model,
    ):
        PooledReturnRegression(
            [b, a], lambda prefix: prefix[-3:], min_train_history=3
        ).predict_returns()
    expected_features = []
    expected_targets = []
    for origin in range(3, 8):
        for offset in (0, 10):
            expected_features.append((np.arange(origin - 2, origin + 1) + offset) / 100)
            expected_targets.append((origin + 1 + offset) / 100)
    np.testing.assert_allclose(model.features, expected_features)
    np.testing.assert_allclose(model.targets, expected_targets)
    np.testing.assert_allclose(model.current, [[0.06, 0.07, 0.08], [0.16, 0.17, 0.18]])


@pytest.mark.parametrize("signals", ["ewm", "12-2"])
def test_calibration_is_not_raw_cumulative_return_and_is_order_independent(signals):
    assets = [stock("B"), stock("A", np.linspace(-0.05, 0.1, 40))]
    results = CalibratedMomentumPredictionStrategy(
        assets, signals=signals
    ).predict_returns()
    reverse = CalibratedMomentumPredictionStrategy(
        assets[::-1], signals=signals
    ).predict_returns()
    assert results == reverse
    assert np.isfinite(list(results.values())).all()
    assert results["A"] != pytest.approx(
        momentum_12_2(assets[1].data.returns.dropna().to_numpy())
    )


def test_cold_start_falls_back_to_mean_without_fitting_scaler():
    asset = stock(returns=np.full(12, 0.02))
    with patch(
        "investment_tool.strategies.prediction_strategies.pooled_return_regression.make_pipeline"
    ) as fit:
        assert CalibratedMomentumPredictionStrategy([asset]).predict_returns()[
            "TEST"
        ] == pytest.approx(0.02)
        fit.assert_not_called()
