"""Exponentially-weighted average prediction strategy.

Predicts expected returns using an exponentially-weighted mean of
historical returns, giving higher weight to recent observations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence


from investment_tool.defaults import PREDICTION
from investment_tool.domain.model_data.entities.stock import Stock
from .base_prediction_strategy import BasePredictionStrategy

logger = logging.getLogger(__name__)


class AveragePredictionStrategy(BasePredictionStrategy):
    """Predict next-period returns as the exponentially-weighted mean.

    Uses ``pd.Series.ewm(span=span).mean()`` so that recent regime
    shifts are reflected faster than a flat arithmetic mean.

    Parameters
    ----------
    stocks : Sequence[Stock]
        Stocks with ``data["returns"]`` column.
    span : int, optional
        EWM span (in periods).  Default is 6, which gives a half-life
        of about 2.06 months for monthly data.
    """

    def __init__(
        self,
        stocks: Sequence[Stock],
        span: int = PREDICTION.ewm_span,
    ) -> None:
        self.stocks = stocks
        self.span = span

    def predict_returns(self) -> dict[str, float]:
        """Predict expected returns via exponentially-weighted mean.

        Returns
        -------
        dict[str, float]
            Predicted return per symbol.
        """
        predictions: dict[str, float] = {}
        for stock in self.stocks:
            returns = stock.data["returns"].dropna()
            if returns.empty:
                predictions[stock.symbol] = 0.0
            else:
                ewm_mean = returns.ewm(span=self.span).mean().iloc[-1]
                predictions[stock.symbol] = float(ewm_mean)

        logger.info("Predicted returns (EWM span=%d): %s", self.span, predictions)
        return predictions
