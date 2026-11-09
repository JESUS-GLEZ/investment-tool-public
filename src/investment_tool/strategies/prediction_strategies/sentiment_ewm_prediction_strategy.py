"""Sentiment-tilted EWM prediction strategy with linear-regression calibration.

Predicts next-period returns by combining an EWM base prediction with a
sentiment tilt whose strength is **calibrated via linear regression** on
historical data — rather than using a fixed heuristic constant.

Model
-----
At each evaluation point the strategy fits an OLS regression on all
available historical periods:

    actual_return_t = α + β₁ · ewm_return_t + β₂ · (ewm_return_t × sentiment_t) + ε_t

Then the out-of-sample prediction for the *next* period is:

    predicted_return = α̂ + β̂₁ · ewm_return + β̂₂ · (ewm_return × sentiment)

The interaction term ``ewm_return × sentiment`` captures the idea that
sentiment *scales* the EWM signal: positive sentiment amplifies positive
predictions and dampens negative ones, and vice versa.

This is strictly superior to a fixed ``tilt_strength`` because the
regression learns from realised returns how much (if at all) sentiment
actually improves the EWM forecast, avoiding arbitrary constant choices.

Look-ahead bias prevention
--------------------------
* The regression is trained on periods ``[0, T-1]`` and predicts period
  ``T``.  No future data leaks into the fit.
* Sentiment scores are filtered to dates ≤ the evaluation date by the
  caller (benchmark runner or investment strategy).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from investment_tool.constants import StockFields
from investment_tool.defaults import PREDICTION
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.prediction_strategies.base_prediction_strategy import (
    BasePredictionStrategy,
)

logger = logging.getLogger(__name__)

# Minimum number of historical periods required to fit the regression.
# With 3 coefficients (intercept, β₁, β₂) we need at least 4 rows,
# but ≥ 12 gives a more reliable estimate.  Imported from defaults.py.
_MIN_REGRESSION_PERIODS = PREDICTION.min_regression_periods


class SentimentEWMPredictionStrategy(BasePredictionStrategy):
    """EWM + sentiment prediction with OLS-calibrated tilt.

    For each asset independently:

    1. Compute the EWM return series (lookback = ``ewm_span``).
    2. Build a per-period sentiment series from ``sentiment_scores``.
    3. Align the three series: ``actual_return``, ``ewm_return``,
       ``ewm_return × sentiment`` (interaction term).
    4. Fit OLS: ``actual = α + β₁·ewm + β₂·(ewm × sentiment)``.
    5. Predict out-of-sample using the most recent EWM and sentiment
       values.

    When there is insufficient data for the regression (fewer than
    ``_MIN_REGRESSION_PERIODS`` aligned rows), the strategy falls back
    to a plain EWM prediction (no sentiment adjustment).

    Parameters
    ----------
    stocks : Sequence[Stock]
        Stocks with ``data["returns"]`` column.
    sentiment_scores : dict[str, dict[date, float]]
        Sentiment scores per stock and date.
    ewm_span : int, optional
        EWM span in periods (default 3).
    fallback_tilt : float, optional
        Fixed tilt strength used **only** when regression cannot be fitted
        (default 0.0 — i.e. pure EWM fallback).
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        sentiment_scores: dict[str, dict[date, float]],
        ewm_span: int = PREDICTION.ewm_span,
        fallback_tilt: float = PREDICTION.fallback_tilt,
    ) -> None:
        self.stocks = stocks
        self.sentiment_scores = sentiment_scores
        self.ewm_span = ewm_span
        self.fallback_tilt = fallback_tilt

        # Store calibration diagnostics for external inspection
        self.calibration_diagnostics: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_sentiment_series(
        scores: dict[date, float],
        index: pd.DatetimeIndex,
    ) -> pd.Series:
        """Convert per-date sentiment dict to an aligned pandas Series.

        Missing dates are forward-filled (sentiment persists until the
        next observation) then filled with 0.0 (neutral).

        Parameters
        ----------
        scores : dict[date, float]
            Raw sentiment scores keyed by date.
        index : pd.DatetimeIndex
            Target date index to align to.

        Returns
        -------
        pd.Series
            Aligned sentiment series.
        """
        if not scores:
            return pd.Series(0.0, index=index, dtype=float)

        raw = pd.Series(scores, dtype=float)
        raw.index = pd.to_datetime(raw.index).normalize()
        aligned = raw.reindex(index).ffill().fillna(0.0)
        # Clamp to [-1, 1]
        return aligned.clip(-1.0, 1.0)

    def _fit_and_predict_single(
        self,
        symbol: str,
        returns: pd.Series,
        sentiment: pd.Series,
    ) -> tuple[float, dict[str, float]]:
        """Fit OLS for one asset and return the out-of-sample prediction.

        Training rows use **lagged** features (shift=1) so the feature at
        row *t* sees only returns and sentiment up to *t−1*.  The
        prediction uses the **current** (unshifted) EWM and sentiment to
        forecast the *next* unobserved period's return.

        Parameters
        ----------
        symbol : str
            Asset symbol (for logging).
        returns : pd.Series
            Historical returns series.
        sentiment : pd.Series
            Aligned sentiment series.

        Returns
        -------
        tuple[float, dict[str, float]]
            ``(prediction, diagnostics)`` where diagnostics contains
            the fitted coefficients and R².
        """
        # ── Feature construction ──────────────────────────────────────
        #
        # We want the regression to learn:
        #     return_t = f(ewm of returns up to t-1, sentiment up to t-1)
        #
        # Then at prediction time we feed (ewm up to T, sentiment at T)
        # to forecast return at T+1 (the next, unobserved period).
        #
        # Training features: shift(1) so feature at row t sees only data
        # up to t-1.  This prevents the target from leaking into features.

        # Unshifted series — used to build the prediction-time feature
        ewm_unshifted = returns.ewm(span=self.ewm_span).mean()

        # Lagged series — used for training rows
        ewm_lagged = ewm_unshifted.shift(1)
        sentiment_lagged = sentiment.shift(1)
        interaction_lagged = ewm_lagged * sentiment_lagged

        # Build training feature matrix — drop NaN from EWM warm-up + shift
        features_lagged = pd.DataFrame(
            {
                "ewm": ewm_lagged,
                "interaction": interaction_lagged,
            },
            index=returns.index,
        )
        aligned = pd.concat(
            [returns.rename("actual"), features_lagged], axis=1
        ).dropna()

        diagnostics: dict[str, float] = {}

        # ── Prediction-time features (unshifted, current values) ─────
        # These represent "what we know right now" to predict the *next*
        # period's return.
        current_ewm = float(ewm_unshifted.iloc[-1]) if len(ewm_unshifted) > 0 else 0.0
        current_sent = float(sentiment.iloc[-1]) if len(sentiment) > 0 else 0.0
        current_interaction = current_ewm * current_sent

        if len(aligned) < _MIN_REGRESSION_PERIODS:
            # Fallback: plain EWM (or fixed tilt if configured)
            prediction = current_ewm * (1.0 + self.fallback_tilt * current_sent)
            diagnostics = {
                "alpha": 0.0,
                "beta_ewm": 1.0,
                "beta_interaction": self.fallback_tilt,
                "r_squared": float("nan"),
                "n_samples": float(len(aligned)),
                "calibrated": 0.0,
            }
            logger.debug(
                "%s: insufficient data (%d rows) for regression — "
                "using fallback tilt=%.2f",
                symbol,
                len(aligned),
                self.fallback_tilt,
            )
            return prediction, diagnostics

        # Train on ALL aligned rows (features are already lagged by 1,
        # so there's no leakage).  Predict for the *next* unobserved
        # period using current (unshifted) EWM and sentiment.
        X_train = aligned[["ewm", "interaction"]].values
        y_train = aligned["actual"].values
        X_pred = np.array([[current_ewm, current_interaction]])

        model = LinearRegression()
        model.fit(X_train, y_train)

        prediction = float(model.predict(X_pred)[0])

        # Diagnostics
        alpha = float(model.intercept_)
        beta_ewm = float(model.coef_[0])
        beta_interaction = float(model.coef_[1])
        r_squared = float(model.score(X_train, y_train))

        # Derive the effective tilt strength for interpretability:
        # adjusted = α + β_ewm · ewm + β_interaction · (ewm · sent)
        #          = α + ewm · (β_ewm + β_interaction · sent)
        # Comparing with: ewm × (1 + tilt · sent)  →  tilt ≈ β_interaction / β_ewm
        effective_tilt = beta_interaction / beta_ewm if abs(beta_ewm) > 1e-8 else 0.0

        diagnostics = {
            "alpha": alpha,
            "beta_ewm": beta_ewm,
            "beta_interaction": beta_interaction,
            "effective_tilt": effective_tilt,
            "r_squared": r_squared,
            "n_samples": float(len(aligned)),
            "calibrated": 1.0,
        }

        logger.debug(
            "%s: OLS calibrated — α=%.6f, β_ewm=%.4f, β_sent=%.4f, "
            "effective_tilt=%.4f, R²=%.4f (n=%d)",
            symbol,
            alpha,
            beta_ewm,
            beta_interaction,
            effective_tilt,
            r_squared,
            len(aligned),
        )

        return prediction, diagnostics

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_returns(self) -> dict[str, float]:
        """Predict next-period returns with OLS-calibrated sentiment tilt.

        For each stock, fits a per-asset linear regression on historical
        EWM returns and sentiment interaction, then predicts
        out-of-sample.

        Returns
        -------
        dict[str, float]
            Predicted return per symbol.
        """
        predictions: dict[str, float] = {}
        self.calibration_diagnostics = {}

        for stock in self.stocks:
            returns = stock.data[StockFields.RETURNS].dropna()
            if returns.empty:
                predictions[stock.symbol] = 0.0
                continue

            # Normalise index for alignment
            returns.index = pd.to_datetime(returns.index).normalize()

            # Build aligned sentiment series
            stock_sentiments = self.sentiment_scores.get(stock.symbol, {})
            sentiment = self._build_sentiment_series(
                stock_sentiments, pd.DatetimeIndex(returns.index)
            )

            prediction, diagnostics = self._fit_and_predict_single(
                stock.symbol, returns, sentiment
            )

            predictions[stock.symbol] = prediction
            self.calibration_diagnostics[stock.symbol] = diagnostics

        # Log summary
        calibrated_count = sum(
            1
            for d in self.calibration_diagnostics.values()
            if d.get("calibrated", 0.0) > 0.5
        )
        total_count = len(self.calibration_diagnostics)
        avg_r2 = (
            np.nanmean(
                [
                    d["r_squared"]
                    for d in self.calibration_diagnostics.values()
                    if d.get("calibrated", 0.0) > 0.5
                ]
            )
            if calibrated_count > 0
            else float("nan")
        )

        logger.info(
            "SentimentEWM prediction complete: %d/%d assets calibrated, " "avg R²=%.4f",
            calibrated_count,
            total_count,
            avg_r2,
        )

        return predictions
