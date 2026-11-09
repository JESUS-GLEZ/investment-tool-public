"""Ridge VARX return prediction strategy.

Implements a vector autoregressive model with exogenous features
(moving averages and sentiment scores) using Ridge regression.  A single
multi-output fit predicts all assets simultaneously, avoiding N
redundant matrix inversions.

All features are **standardised** before fitting so that Ridge
regularisation penalises coefficients uniformly regardless of the
original feature scale.  The moving-average feature is expressed as a
price-to-MA *ratio* (centred on 1.0) rather than a raw price level,
preventing large absolute prices from dominating the regression.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from investment_tool.constants import StockFields
from investment_tool.defaults import PREDICTION
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.base_prediction_strategy import (
    BasePredictionStrategy,
)
from investment_tool.utils import recently_observed_symbols

logger = logging.getLogger(__name__)


class RidgeVARXPredictionStrategy(BasePredictionStrategy):
    """Ridge-regularised VARX model for multi-asset return prediction.

    Features are constructed from:

    * **Lagged returns** (endogenous): ``lag_order`` lags of each asset's
      period returns.
    * **Price / MA ratio** (exogenous): Close / MA-5, a unit-free momentum
      proxy centred around 1.0.
    * **Sentiment score** (exogenous): per-asset sentiment from
      transformer-based news analysis (range ≈ −1 to 1).

    All features are **z-score standardised** before fitting so that Ridge
    penalises every coefficient uniformly. Features available at the end of
    period t predict the return of period t+1; callers supply completed history.

    Parameters
    ----------
    stocks : Sequence[Stock]
        Stock entities with computed technical indicators.
    stock_sentiment_scores : dict[str, dict[date, float]]
        Sentiment scores per stock and date.
    lag_order : int, optional
        Number of lagged return periods to include, by default 3.
    alpha : float, optional
        Ridge regularisation strength, by default 10.0.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        stock_sentiment_scores: dict[str, dict[date, float]],
        lag_order: int = PREDICTION.lag_order,
        alpha: float = PREDICTION.ridge_alpha,
    ) -> None:
        self.stocks = stocks
        self.stock_sentiment_scores = stock_sentiment_scores
        self.lag_order = lag_order
        self.alpha = alpha

    def _prepare_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare endogenous and exogenous feature matrices.

        Endogenous variables are each asset's period returns.  Exogenous
        variables include the **price / MA-5 ratio** (a scale-free momentum
        signal) and sentiment scores.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(returns_df, exog_df)`` — returns matrix and exogenous
            features matrix, both indexed by normalised datetime.
        """
        # Build returns matrix (periods × assets)
        returns_df = pd.concat(
            [stock.data[StockFields.RETURNS] for stock in self.stocks], axis=1
        )
        returns_df.columns = [stock.symbol for stock in self.stocks]
        returns_df.index = pd.to_datetime(returns_df.index).normalize()
        # Forward-fill only to avoid look-ahead bias; drop remaining leading NaN
        eligible = recently_observed_symbols(returns_df)
        if not eligible:
            raise ValueError("No assets have enough recent return history for VARX")
        returns_df = returns_df[eligible].ffill().dropna()

        exog_data: dict[str, pd.Series] = {}
        for stock in self.stocks:
            if stock.symbol not in eligible:
                continue
            # Price / MA-5 ratio — unit-free momentum proxy centred on 1.0.
            # Using the ratio (instead of the raw MA price) avoids injecting
            # features that are orders of magnitude larger than returns, which
            # would cause Ridge to produce amplified predictions.
            ma_col_name = f"{StockFields.MOVING_AVERAGE}_5"
            if ma_col_name in stock.data.columns:
                close = pd.to_numeric(stock.data[StockFields.CLOSE], errors="coerce")
                ma5 = pd.to_numeric(stock.data[ma_col_name], errors="coerce")
                price_ma_ratio = (close / ma5).replace([np.inf, -np.inf], np.nan)
                price_ma_ratio.index = pd.to_datetime(price_ma_ratio.index).normalize()
                price_ma_ratio = price_ma_ratio.reindex(returns_df.index).ffill()
            else:
                # Fallback: neutral ratio
                price_ma_ratio = pd.Series(1.0, index=returns_df.index)

            exog_data[f"{stock.symbol}_price_ma_ratio"] = price_ma_ratio

            # Sentiment score
            sent_col = f"{stock.symbol}_{StockFields.SENTIMENT_SCORE}"
            sent_series = pd.Series(
                self.stock_sentiment_scores.get(stock.symbol, {}), dtype=float
            )
            sent_series.index = pd.to_datetime(sent_series.index)
            sent_series = (
                pd.to_numeric(sent_series, errors="coerce")
                .reindex(returns_df.index)
                .ffill()
                .fillna(0.0)  # Sentiment defaults to neutral when absent
            )
            exog_data[sent_col] = sent_series

        exog_df = pd.DataFrame(exog_data, index=returns_df.index)
        return returns_df, exog_df

    def predict_returns(self) -> dict[str, float]:
        """Execute the Ridge VARX prediction strategy.

        Builds the lagged endogenous matrix, combines it with exogenous
        data, and pairs each completed period's features with the following
        observed return. Standardises using training rows only, trains a single
        multi-output Ridge model, and forecasts the next unobserved period
        using the final completed period's features.

        Returns
        -------
        dict[str, float]
            Predicted next-period return per symbol.
        """
        returns_df, exog_df = self._prepare_data()

        # Index features by the completed period where they become available.
        # For a t+1 target, lag 1 is r(t), lag 2 is r(t-1), etc.
        X_lags = []
        for lag in range(self.lag_order):
            X_lags.append(returns_df.shift(lag))
        X_lags_df = pd.concat(X_lags, axis=1)

        X_combined = pd.concat([X_lags_df, exog_df], axis=1)
        y_next = returns_df.shift(-1)
        training_rows = X_combined.notna().all(axis=1) & y_next.notna().all(axis=1)
        # Keep the actual final row, even if incomplete. Dropping it would
        # silently change the forecast horizon to an older period.
        X_forecast = X_combined.iloc[-1:]

        expected_returns: dict[str, float] = {}
        if not training_rows.any() or X_forecast.empty or X_forecast.isna().any().any():
            logger.warning(
                "Insufficient data for a next-period forecast: need a completed "
                "training pair and complete latest-period features. "
                "Returning zero predictions for all assets."
            )
            for symbol in returns_df.columns:
                expected_returns[symbol] = 0.0
            return expected_returns

        # The latest observed return is a training target for the previous
        # origin. The final origin has no observed target and is forecast-only.
        X_train = X_combined.loc[training_rows].values
        X_pred = X_forecast.values
        y_train = y_next.loc[training_rows].values  # (n_samples, n_assets)

        # Standardise features so Ridge penalises all coefficients uniformly
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_pred_scaled = scaler.transform(X_pred)

        # Single multi-output fit — mathematically identical to N separate fits
        # for Ridge (closed-form), but avoids N separate matrix inversions
        model = Ridge(alpha=self.alpha)
        model.fit(X_train_scaled, y_train)
        raw_pred = model.predict(X_pred_scaled)[0]
        # When there's only 1 asset, predict returns a scalar; ensure 1-D array
        predictions = np.atleast_1d(raw_pred)

        for i, symbol in enumerate(returns_df.columns):
            expected_returns[symbol] = float(predictions[i])

        return expected_returns
