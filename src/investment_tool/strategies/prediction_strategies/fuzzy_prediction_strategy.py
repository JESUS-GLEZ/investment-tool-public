"""Research predictors reconstructed from González's 2019 fuzzy-series thesis.

Kernels use percentage-point returns; the public strategy contract uses decimals.
Each feature row is constructed from its own historical prefix. These predictors
are comparison options, not investment-strategy defaults.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from investment_tool.domain.model_data.entities.stock import Stock
from .base_prediction_strategy import BasePredictionStrategy
from .pooled_return_regression import PooledReturnRegression, contiguous_returns


@dataclass(frozen=True)
class FuzzySettings:
    """One shared formulation; beta=0 disables frequency-based width adaptation."""

    partitions: int = 9
    lags: int = 3
    sigma: float = 10.0
    beta: float = 2.0
    trend: bool = True
    grid_points: int = 513

    def __post_init__(self) -> None:
        if self.partitions < 2 or self.lags < 1 or self.grid_points < 3:
            raise ValueError("Require >=2 partitions, >=1 lag and >=3 grid points")
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")


def trend_correction(returns_pct: NDArray[np.float64]) -> float:
    """Equation 4.10: mean of past return minus each subsequent lag average.

    Uses lag j, not the experiment counter q in the printed MATLAB listing.
    """
    if len(returns_pct) < 2:
        return 0.0
    reverse = returns_pct[::-1]
    later_means = np.cumsum(reverse)[:-1] / np.arange(1, len(reverse))
    return float(np.mean(reverse[1:] - later_means))


def _kernels(
    returns_pct: NDArray[np.float64], settings: FuzzySettings
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], float, float]:
    low, high = float(np.min(returns_pct)), float(np.max(returns_pct))
    # Appendix B.3 uses abs(Dmax**2/100) on both sides.
    margin = abs(high**2 / 100.0)
    lower, upper = low - margin, high + margin
    centers = lower + (np.arange(settings.partitions) + 0.5) * (
        (upper - lower) / settings.partitions
    )
    # With equal baseline widths, argmax membership is the nearest center.
    states = np.argmin(np.abs(returns_pct[:, None] - centers), axis=1)
    frequency = np.bincount(states, minlength=settings.partitions) / len(returns_pct)
    widths = settings.sigma + settings.beta * frequency
    return centers, widths, states.astype(np.int64), lower, upper


def fuzzy_forecast(values: NDArray[np.float64], settings: FuzzySettings) -> float:
    """Max-aggregate recent states, centroid on U, optional Favg, decimal output."""
    history = contiguous_returns(values) * 100.0
    if len(history) < settings.lags:
        return float(np.mean(history) / 100.0) if len(history) else float("nan")
    centers, widths, states, lower, upper = _kernels(history, settings)
    if upper == lower:
        centroid = lower
    else:
        selected = np.unique(states[-settings.lags :])
        # Resolve narrow kernels even when an extreme return makes U very wide.
        local_grid = centers[selected, None] + widths[selected, None] * np.linspace(
            -8.0, 8.0, 129
        )
        grid = np.unique(
            np.r_[np.linspace(lower, upper, settings.grid_points), local_grid.ravel()]
        )
        grid = grid[(grid >= lower) & (grid <= upper)]
        logs = -0.5 * (
            (grid[:, None] - centers[selected]) / widths[selected]
        ) ** 2 - np.log(widths[selected])
        aggregate = np.max(logs, axis=1)
        membership = np.exp(aggregate - np.max(aggregate))
        # Common log scaling cancels in the centroid and prevents underflow.
        centroid = float(np.trapz(grid * membership, grid) / np.trapz(membership, grid))
    correction = trend_correction(history) if settings.trend else 0.0
    return (centroid + correction) / 100.0


def fuzzy_features(
    values: NDArray[np.float64], settings: FuzzySettings
) -> NDArray[np.float64]:
    """Soft state memberships of recent returns plus historical mean/volatility.

    Unlike the thesis max aggregation, separate lag rows preserve their order.
    Centers and widths are rebuilt using this prefix only, never its target.
    """
    history = contiguous_returns(values)
    if len(history) < settings.lags:
        raise ValueError("Insufficient contiguous history for fuzzy features")
    centers, widths, _, _, _ = _kernels(history * 100.0, settings)
    logs = -0.5 * (
        (history[-settings.lags :, None] * 100.0 - centers) / widths
    ) ** 2 - np.log(widths)
    memberships = np.exp(logs - np.max(logs, axis=1, keepdims=True))
    memberships /= np.sum(memberships, axis=1, keepdims=True)
    return np.concatenate(
        (memberships.ravel(), [float(np.mean(history)), float(np.std(history))])
    )


class ReturnBaselinePredictionStrategy(BasePredictionStrategy):
    """Zero return (unchanged price) or expanding historical arithmetic mean."""

    def __init__(self, stocks: Sequence[Stock], *, zero: bool = False) -> None:
        self.stocks = stocks
        self.zero = zero

    def predict_returns(self) -> dict[str, float]:
        predictions: dict[str, float] = {}
        for stock in self.stocks:
            values = stock.data["returns"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            predictions[stock.symbol] = (
                0.0 if self.zero else float(np.mean(finite)) if len(finite) else 0.0
            )
        return predictions


class FuzzyPredictionStrategy(BasePredictionStrategy):
    """Direct thesis reconstruction, including an optional trend correction."""

    def __init__(
        self, stocks: Sequence[Stock], settings: FuzzySettings = FuzzySettings()
    ) -> None:
        self.stocks = stocks
        self.settings = settings

    def predict_returns(self) -> dict[str, float]:
        return {
            stock.symbol: fuzzy_forecast(
                stock.data["returns"].to_numpy(dtype=float), self.settings
            )
            for stock in self.stocks
        }


class PooledRidgePredictionStrategy(PooledReturnRegression):
    """One regularised next-return regression, pooled across supplied assets.

    Training features at t use only returns through t; their label is t+1.
    No hyperparameter search. The raw-lag control shares the same training rows,
    scaler, alpha and fallback as the fuzzy-feature versions.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        settings: FuzzySettings = FuzzySettings(),
        *,
        fuzzy: bool = True,
        min_train_history: int = 12,
        alpha: float = 10.0,
    ) -> None:
        if min_train_history < settings.lags or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Require history >= lags and finite positive alpha")
        self.settings = settings
        self.fuzzy = fuzzy
        super().__init__(
            stocks, self._features, min_train_history=min_train_history, alpha=alpha
        )

    def _features(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.fuzzy:
            return fuzzy_features(values, self.settings)
        return np.concatenate(
            (values[-self.settings.lags :], [np.mean(values), np.std(values)])
        )
