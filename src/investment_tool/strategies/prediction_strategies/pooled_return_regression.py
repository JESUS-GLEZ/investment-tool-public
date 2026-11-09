"""Shared causal next-return regression for research predictor configurations."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from investment_tool.domain.model_data.entities.stock import Stock
from .base_prediction_strategy import BasePredictionStrategy


def contiguous_returns(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Trailing finite block: missing periods must not become adjacent lags."""
    missing = np.flatnonzero(~np.isfinite(values))
    start = int(missing[-1]) + 1 if len(missing) else 0
    return values[start:]


class PooledReturnRegression(BasePredictionStrategy):
    """Pool historical prefix/next-return pairs; fit scaler on training rows only.

    The benchmark calls this with availability-filtered history at each origin.
    No random split or parameter tuning occurs; target months advance together.
    Missing periods restart feature history. Cold starts use the asset mean.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        feature_builder: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        *,
        min_train_history: int = 12,
        alpha: float = 10.0,
    ) -> None:
        if min_train_history < 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Require positive history and finite positive alpha")
        self.stocks = stocks
        self.feature_builder = feature_builder
        self.min_train_history = min_train_history
        self.alpha = alpha

    def predict_returns(self) -> dict[str, float]:
        rows: list[NDArray[np.float64]] = []
        targets: list[float] = []
        target_dates: list[pd.Timestamp] = []
        current: dict[str, NDArray[np.float64]] = {}
        predictions: dict[str, float] = {}
        # Stable row order also makes results independent of universe-file order.
        for stock in sorted(self.stocks, key=lambda item: item.symbol):
            values = stock.data["returns"].to_numpy(dtype=float)
            history = contiguous_returns(values)
            predictions[stock.symbol] = (
                float(np.mean(history)) if len(history) else float("nan")
            )
            if len(history) >= self.min_train_history:
                current[stock.symbol] = self.feature_builder(history)
            # Restart after gaps; never pair nonadjacent calendar observations.
            block_start = 0
            for origin in range(len(values) - 1):
                if not np.isfinite(values[origin]):
                    block_start = origin + 1
                    continue
                prefix = values[block_start : origin + 1]
                target = values[origin + 1]
                if len(prefix) >= self.min_train_history and np.isfinite(target):
                    rows.append(self.feature_builder(prefix))
                    targets.append(float(target))
                    target_dates.append(pd.Timestamp(stock.data.index[origin + 1]))
        if len(rows) < self.min_train_history or not current:
            return predictions
        model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
        # Chronological batches keep all assets of a target month together.
        # Every label is already historical at the caller's forecast origin.
        order = np.argsort(
            np.asarray(target_dates, dtype="datetime64[ns]"), kind="stable"
        )
        model.fit(np.asarray(rows)[order], np.asarray(targets)[order])
        symbols = list(current)
        forecasts = model.predict(np.asarray([current[symbol] for symbol in symbols]))
        predictions.update(zip(symbols, map(float, forecasts), strict=True))
        return predictions
