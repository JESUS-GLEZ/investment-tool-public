"""Momentum–Risk-Parity investment strategy.

Combines cross-sectional and time-series momentum with inverse-volatility
(risk-parity) weighting.  This approach avoids explicit return prediction
— the most error-prone step in quantitative investing — and instead
relies on two of the most robust empirical findings in finance:

1. **Momentum**: assets that have performed well recently tend to
   continue outperforming in the near term.
2. **Risk parity**: allocating based on inverse realised volatility
   equalises the *risk contribution* of each position, producing
   more diversified portfolios than market-cap or equal weighting.

Pipeline
--------
1. Compute trailing total return over ``lookback`` periods (momentum score).
2. Apply a **trend filter**: exclude assets whose price is below their
   ``trend_window``-period simple moving average (time-series momentum).
3. Rank surviving assets by momentum score and keep the top ``top_n``.
4. Weight the selected assets by inverse realised volatility
   (``vol_window``-period rolling standard deviation of returns).
5. Normalise weights to sum to 1.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from investment_tool.constants import StockFields
from investment_tool.defaults import MOMENTUM
from investment_tool.strategies.orchestrators.base_investment_strategy import (
    BaseInvestmentStrategy,
)
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.news import News
from investment_tool.domain.model_data.entities.stock import Stock

logger = logging.getLogger(__name__)


class InvestmentStrategyMomentumRiskParity(BaseInvestmentStrategy):
    """Momentum + inverse-volatility risk-parity strategy.

    Parameters
    ----------
    instance : Instance
        Execution instance with trading-period references.
    stocks : Sequence[Stock]
        Stock entities (full, unfiltered data — filtering is handled by
        the base class per trading period).
    news : dict[tuple[str, date], News] | None
        News data (unused by this strategy but accepted for interface
        compatibility).
    top_n : int, optional
        Number of assets to hold after momentum ranking, by default 10.
    lookback : int, optional
        Trailing periods for momentum score, by default 6 (months).
    vol_window : int, optional
        Rolling window for volatility estimation, by default 6 (months).
    trend_window : int, optional
        SMA window for the trend filter, by default 10 (months).
    min_weight : float, optional
        Minimum position weight; assets below this are zeroed out and
        weight is redistributed, by default 0.02.
    output_directory : Path | None, optional
        Directory for output artefacts.

    Notes
    -----
    Default parameters are tuned for **monthly** rebalancing.  For other
    frequencies adjust ``lookback``, ``vol_window``, and ``trend_window``
    accordingly (e.g. multiply by 4 for weekly, by 21 for daily).
    """

    def __init__(
        self,
        instance: Instance,
        stocks: Sequence[Stock],
        news: dict[tuple[str, date], News] | None = None,
        *,
        top_n: int = MOMENTUM.top_n,
        lookback: int = MOMENTUM.lookback,
        vol_window: int = MOMENTUM.vol_window,
        trend_window: int = MOMENTUM.trend_window,
        min_weight: float = MOMENTUM.min_weight,
        output_directory: Path | None = None,
    ) -> None:
        super().__init__(instance=instance, stocks=stocks, news=news)
        self._instance = instance
        self.top_n = top_n
        self.lookback = lookback
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.min_weight = min_weight
        self.output_directory = output_directory

        logger.info(
            "Initialized MomentumRiskParity strategy "
            f"(top_n={top_n}, lookback={lookback}, "
            f"vol_window={vol_window}, trend_window={trend_window})"
        )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _build_data_matrices(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build returns and close-price matrices from filtered stocks.

        Only stocks with sufficient history for both the momentum lookback
        and the trend-filter window are included.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            ``(returns_df, close_df)`` — DataFrames indexed by date with
            one column per qualifying stock.

        Raises
        ------
        RuntimeError
            If no stocks have enough history.
        """
        min_history = max(self.lookback, self.trend_window)

        returns_map: dict[str, pd.Series] = {}
        close_map: dict[str, pd.Series] = {}
        for stock in self.stocks:
            r = stock.data[StockFields.RETURNS].dropna()
            c = stock.data[StockFields.CLOSE].dropna()
            if len(r) >= self.lookback and len(c) >= self.trend_window:
                returns_map[stock.symbol] = r
                close_map[stock.symbol] = c

        if not returns_map:
            raise RuntimeError(
                f"No stocks have sufficient history (need ≥ {min_history} periods)"
            )

        return pd.DataFrame(returns_map), pd.DataFrame(close_map)

    def _select_assets(
        self,
        returns_df: pd.DataFrame,
        close_df: pd.DataFrame,
    ) -> list[str]:
        """Select top momentum assets that pass the trend filter.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns matrix (periods × assets).
        close_df : pd.DataFrame
            Close prices (periods × assets).

        Returns
        -------
        list[str]
            Symbols of selected assets (at most ``top_n``).
        """
        momentum = self._compute_momentum(returns_df)

        # Trend filter: keep only assets whose last close > SMA
        trending_up = self._apply_trend_filter(close_df)
        filtered_momentum = momentum[momentum.index.isin(trending_up)]

        if filtered_momentum.empty:
            logger.warning(
                "All assets filtered out by trend filter — "
                "falling back to top momentum stocks without filter"
            )
            filtered_momentum = momentum

        n_select = min(self.top_n, len(filtered_momentum))
        selected = filtered_momentum.nlargest(n_select).index.tolist()
        logger.info(
            "Selected %d assets by momentum: %s",
            len(selected),
            ", ".join(selected),
        )
        return selected

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_strategy(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.Series:
        """Execute the momentum--risk-parity pipeline.

        Orchestrates:

        1. Build returns and close-price matrices.
        2. Select top-momentum assets passing the trend filter.
        3. Weight by inverse realised volatility.
        4. Clean dust positions and normalise.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Accepted for interface compatibility but unused —
            this strategy has no long-running optimisation loop.

        Returns
        -------
        pd.Series
            Normalised portfolio weights indexed by stock symbol.

        Raises
        ------
        RuntimeError
            If no assets survive the filtering steps.
        """
        logger.info("Running momentum–risk-parity strategy")

        # 1. Data → 2. Selection → 3. Weighting → 4. Cleanup
        returns_df, close_df = self._build_data_matrices()
        selected = self._select_assets(returns_df, close_df)
        weights = self._inverse_vol_weights(returns_df, selected)
        weights = self._clean_weights(weights)

        self._log_weights(weights)
        return weights

    @staticmethod
    def _log_weights(weights: pd.Series) -> None:
        """Log selected portfolio weights (non-dust positions only)."""
        active = {s: w for s, w in weights.items() if w >= 0.01}
        logger.info(
            "Final weights (%d assets): %s",
            len(active),
            ", ".join(f"{s}:{w:.4f}" for s, w in active.items()),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_momentum(self, returns_df: pd.DataFrame) -> pd.Series:
        """Compute trailing total return as the momentum score.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns matrix (periods × assets).

        Returns
        -------
        pd.Series
            Momentum score per asset (higher = stronger momentum).
        """
        trailing = returns_df.iloc[-self.lookback :]
        # Compound return over the lookback window
        cum_return: pd.Series = (1 + trailing).prod() - 1
        return cum_return.sort_values(ascending=False)

    def _apply_trend_filter(self, close_df: pd.DataFrame) -> list[str]:
        """Keep only assets whose last close is above their SMA.

        Parameters
        ----------
        close_df : pd.DataFrame
            Close prices (periods × assets).

        Returns
        -------
        list[str]
            Symbols of assets in an uptrend.
        """
        sma = close_df.rolling(self.trend_window).mean().iloc[-1]
        last_close = close_df.iloc[-1]
        above = last_close > sma
        trending = above[above].index.tolist()
        logger.debug(
            f"Trend filter: {len(trending)}/{len(close_df.columns)} assets "
            f"above {self.trend_window}-period SMA"
        )
        return trending

    def _inverse_vol_weights(
        self, returns_df: pd.DataFrame, symbols: list[str]
    ) -> pd.Series:
        """Compute inverse-volatility weights for *symbols*.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Full returns matrix.
        symbols : list[str]
            Selected asset symbols.

        Returns
        -------
        pd.Series
            Normalised weights indexed by all symbols in ``returns_df``
            (unselected assets have weight 0).
        """
        vol = returns_df[symbols].iloc[-self.vol_window :].std()
        # Guard against zero-vol assets
        vol = vol.replace(0.0, np.nan).dropna()
        if vol.empty:
            # Fallback: equal weight
            inv_vol = pd.Series(1.0, index=symbols)
        else:
            inv_vol = 1.0 / vol

        raw = inv_vol / inv_vol.sum()
        weights = pd.Series(0.0, index=returns_df.columns)
        weights[raw.index] = raw.values
        return weights

    def _clean_weights(self, weights: pd.Series) -> pd.Series:
        """Zero out dust positions and renormalise.

        Parameters
        ----------
        weights : pd.Series
            Raw weights.

        Returns
        -------
        pd.Series
            Cleaned, normalised weights summing to 1.
        """
        weights[weights < self.min_weight] = 0.0
        total = float(weights.sum())
        if total <= 0:
            n = len(weights)
            return pd.Series(1.0 / n, index=weights.index)
        return pd.Series((weights / total).values, index=weights.index)
