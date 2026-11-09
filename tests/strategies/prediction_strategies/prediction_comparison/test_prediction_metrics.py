"""Unit tests for prediction_metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_metrics import (
    compute_all_prediction_metrics,
    compute_per_asset_metrics,
    directional_accuracy,
    hit_rate_top_k,
    information_coefficient,
    mean_absolute_error,
    mean_signed_error,
    root_mean_squared_error,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_aligned(
    n_dates: int = 10,
    n_assets: int = 5,
    noise: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aligned predictions and actuals with controllable noise."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="ME")
    symbols = [f"S{i}" for i in range(n_assets)]
    actuals = pd.DataFrame(
        np.random.randn(n_dates, n_assets) * 0.05,
        index=dates,
        columns=symbols,
    )
    predictions = actuals + np.random.randn(n_dates, n_assets) * noise
    return predictions, actuals


def _make_perfect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predictions == actuals."""
    pred, act = _make_aligned(noise=0.0)
    return pred, act


def _make_empty() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.DataFrame(), pd.DataFrame()


# ── MAE Tests ────────────────────────────────────────────────────────────────


class TestMAE:
    def test_zero_for_perfect(self) -> None:
        pred, act = _make_perfect()
        assert mean_absolute_error(pred, act) == pytest.approx(0.0, abs=1e-10)

    def test_positive_for_noisy(self) -> None:
        pred, act = _make_aligned(noise=0.05)
        assert mean_absolute_error(pred, act) > 0.0

    def test_nan_for_empty(self) -> None:
        pred, act = _make_empty()
        assert np.isnan(mean_absolute_error(pred, act))


# ── RMSE Tests ───────────────────────────────────────────────────────────────


class TestRMSE:
    def test_zero_for_perfect(self) -> None:
        pred, act = _make_perfect()
        assert root_mean_squared_error(pred, act) == pytest.approx(0.0, abs=1e-10)

    def test_ge_mae(self) -> None:
        pred, act = _make_aligned(noise=0.05)
        assert root_mean_squared_error(pred, act) >= mean_absolute_error(pred, act)


# ── Directional Accuracy Tests ───────────────────────────────────────────────


class TestDirectionalAccuracy:
    def test_perfect_direction(self) -> None:
        pred, act = _make_perfect()
        assert directional_accuracy(pred, act) == pytest.approx(1.0, abs=1e-10)

    def test_between_zero_and_one(self) -> None:
        pred, act = _make_aligned(noise=0.1)
        da = directional_accuracy(pred, act)
        assert 0.0 <= da <= 1.0

    def test_nan_for_empty(self) -> None:
        pred, act = _make_empty()
        assert np.isnan(directional_accuracy(pred, act))

    def test_inverted_predictions(self) -> None:
        """All signs wrong → low accuracy."""
        pred, act = _make_perfect()
        da = directional_accuracy(-pred, act)
        # Most should be wrong (except zeros)
        assert da < 0.2


# ── Information Coefficient Tests ────────────────────────────────────────────


class TestIC:
    def test_perfect_ic(self) -> None:
        pred, act = _make_perfect()
        ic = information_coefficient(pred, act)
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_anti_correlated(self) -> None:
        pred, act = _make_perfect()
        ic = information_coefficient(-pred, act)
        assert ic < 0.0

    def test_nan_for_empty(self) -> None:
        pred, act = _make_empty()
        assert np.isnan(information_coefficient(pred, act))


# ── Bias Tests ───────────────────────────────────────────────────────────────


class TestBias:
    def test_zero_for_perfect(self) -> None:
        pred, act = _make_perfect()
        assert mean_signed_error(pred, act) == pytest.approx(0.0, abs=1e-10)

    def test_positive_bias(self) -> None:
        pred, act = _make_perfect()
        assert mean_signed_error(pred + 0.01, act) > 0.0

    def test_negative_bias(self) -> None:
        pred, act = _make_perfect()
        assert mean_signed_error(pred - 0.01, act) < 0.0


# ── Hit Rate Tests ───────────────────────────────────────────────────────────


class TestHitRate:
    def test_perfect_hit_rate(self) -> None:
        pred, act = _make_perfect()
        assert hit_rate_top_k(pred, act, k=3) == pytest.approx(1.0, abs=1e-10)

    def test_between_zero_and_one(self) -> None:
        pred, act = _make_aligned(noise=0.1)
        hr = hit_rate_top_k(pred, act, k=3)
        assert 0.0 <= hr <= 1.0

    def test_nan_when_k_too_large(self) -> None:
        pred, act = _make_aligned(n_assets=2)
        assert np.isnan(hit_rate_top_k(pred, act, k=5))


# ── compute_all_prediction_metrics ───────────────────────────────────────────


class TestComputeAll:
    def test_returns_all_keys(self) -> None:
        pred, act = _make_aligned()
        metrics = compute_all_prediction_metrics(pred, act)
        expected = {
            "mae",
            "rmse",
            "directional_accuracy",
            "information_coefficient",
            "bias",
            "hit_rate_top_k",
        }
        assert set(metrics.keys()) == expected

    def test_all_finite_for_normal_data(self) -> None:
        pred, act = _make_aligned()
        metrics = compute_all_prediction_metrics(pred, act)
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


# ── compute_per_asset_metrics ────────────────────────────────────────────────


class TestPerAsset:
    def test_one_row_per_asset(self) -> None:
        pred, act = _make_aligned(n_assets=4)
        per = compute_per_asset_metrics(pred, act)
        assert len(per) == 4

    def test_expected_columns(self) -> None:
        pred, act = _make_aligned()
        per = compute_per_asset_metrics(pred, act)
        assert set(per.columns) == {"mae", "rmse", "directional_accuracy", "bias"}

    def test_empty_for_empty_input(self) -> None:
        pred, act = _make_empty()
        per = compute_per_asset_metrics(pred, act)
        assert per.empty


def test_zero_return_baseline_has_no_arbitrary_asset_ranking():
    pred, actual = _make_aligned()
    pred[:] = 0.0
    assert np.isnan(hit_rate_top_k(pred, actual, k=3))
    assert np.isnan(information_coefficient(pred, actual))
    assert np.isfinite(mean_absolute_error(pred, actual))
