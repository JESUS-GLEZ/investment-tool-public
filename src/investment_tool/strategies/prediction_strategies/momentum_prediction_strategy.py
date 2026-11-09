"""Fixed mean shrinkage and calibrated monthly momentum research predictors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from investment_tool.domain.model_data.entities.stock import Stock
from .average_prediction_strategy import AveragePredictionStrategy
from .base_prediction_strategy import BasePredictionStrategy
from .fuzzy_prediction_strategy import ReturnBaselinePredictionStrategy
from .pooled_return_regression import PooledReturnRegression


class MeanBlendPredictionStrategy(BasePredictionStrategy):
    """Fixed convex blend of EWM and expanding historical arithmetic mean.

    The momentum weight is configured, never fitted on evaluation outcomes.
    Both components reuse existing predictors and their missing-data policies.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        *,
        momentum_weight: float = 0.5,
        span: int = 12,
    ) -> None:
        if not np.isfinite(momentum_weight) or not 0 <= momentum_weight <= 1:
            raise ValueError("momentum_weight must be finite and between 0 and 1")
        if span < 1:
            raise ValueError("span must be positive")
        self.stocks = stocks
        self.momentum_weight = momentum_weight
        self.span = span

    def predict_returns(self) -> dict[str, float]:
        mean = ReturnBaselinePredictionStrategy(self.stocks).predict_returns()
        if self.momentum_weight == 0:
            return mean
        ewm = AveragePredictionStrategy(self.stocks, span=self.span).predict_returns()
        if self.momentum_weight == 1:
            return ewm
        return {
            symbol: self.momentum_weight * ewm[symbol]
            + (1 - self.momentum_weight) * value
            for symbol, value in mean.items()
        }


def _ewm_last(values: NDArray[np.float64], span: int) -> float:
    """Equivalent to pandas ewm(span=span, adjust=True) on a finite prefix."""
    weights = (1 - 2 / (span + 1)) ** np.arange(len(values) - 1, -1, -1)
    return float(np.dot(weights, values) / weights.sum())


def momentum_12_2(values: NDArray[np.float64]) -> float:
    """Compound 11 monthly returns, excluding the latest observed month.

    At origin t predicting t+1, these are t-11 through t-1: months 12 to 2
    before the target. The signal is not itself a next-month return forecast.
    """
    if len(values) < 12 or not np.isfinite(values[-12:]).all():
        raise ValueError("12-2 momentum requires 12 contiguous monthly returns")
    return float(np.prod(1 + values[-12:-1]) - 1)


class CalibratedMomentumPredictionStrategy(PooledReturnRegression):
    """Shared ridge calibration using four signals or just 12-2 momentum.

    EWM model: EWM-6, EWM-12, 12-2 compounded momentum, trailing 12-month
    return standard deviation (ddof=0). Single-signal model isolates 12-2
    momentum calibration. Defaults use alpha=10 and >=12 return history.
    Monthly-only registration prevents confusing months with days or weeks.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        *,
        signals: Literal["ewm", "12-2"] = "ewm",
        alpha: float = 10.0,
        min_train_history: int = 12,
    ) -> None:
        if signals not in ("ewm", "12-2"):
            raise ValueError("Unknown momentum signal configuration")
        if min_train_history < 12:
            raise ValueError("Momentum features require at least 12 history periods")
        self.signals = signals
        super().__init__(
            stocks, self._features, min_train_history=min_train_history, alpha=alpha
        )

    def _features(self, values: NDArray[np.float64]) -> NDArray[np.float64]:
        momentum = momentum_12_2(values)
        if self.signals == "12-2":
            return np.array([momentum], dtype=float)
        return np.array(
            [
                _ewm_last(values, 6),
                _ewm_last(values, 12),
                momentum,
                np.std(values[-12:]),
            ],
            dtype=float,
        )
