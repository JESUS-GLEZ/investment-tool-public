"""Buy-and-hold benchmark investment strategy.

Allocates 100 % of portfolio value to a single asset (default: SPY,
the S&P 500 ETF).  The weight vector is trivially recomputed each
period so that the backtester interface is satisfied, but the
allocation never changes — making this an ideal **benchmark** for
comparing active strategies.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

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


class InvestmentStrategyBuyAndHold(BaseInvestmentStrategy):
    """Allocate the entire portfolio to a single asset.

    Parameters
    ----------
    instance : Instance
        Execution instance configuration.
    stocks : Sequence[Stock]
        Stock entities for the strategy.
    news : dict[tuple[str, date], News] | None
        News data (unused — accepted for interface compatibility).
    asset_symbol : str, optional
        Symbol of the asset to hold, by default ``"SPY"``.
    output_directory : Path | None, optional
        Directory for output files (unused — accepted for interface
        compatibility).

    Raises
    ------
    ValueError
        If *asset_symbol* is not found in the provided *stocks*.
    """

    def __init__(
        self,
        instance: Instance,
        stocks: Sequence[Stock],
        news: dict[tuple[str, date], News] | None = None,
        asset_symbol: str = "SPY",
        output_directory: Path | None = None,
    ) -> None:
        super().__init__(instance=instance, stocks=stocks, news=news)
        self._asset_symbol = asset_symbol

        available = {s.symbol for s in stocks}
        if asset_symbol not in available:
            raise ValueError(
                f"Target asset '{asset_symbol}' not found in provided stocks. "
                f"Available: {sorted(available)}"
            )

        logger.info("Initialized InvestmentStrategyBuyAndHold (asset=%s)", asset_symbol)

    def run_strategy(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> pd.Series:
        """Return 100 % allocation to the target asset.

        Parameters
        ----------
        progress_callback : ProgressCallback | None, optional
            Accepted for interface compatibility but unused —
            this strategy always returns immediately.

        Returns
        -------
        pd.Series
            Weights indexed by symbol — 1.0 for the target asset,
            0.0 for all others.
        """
        symbols = [s.symbol for s in self.stocks]
        weights = pd.Series(0.0, index=symbols)

        if self._asset_symbol in weights.index:
            weights[self._asset_symbol] = 1.0
        else:
            # Fallback: the target may have been filtered out for this
            # period (e.g. no historical data).  Distribute equally.
            logger.warning(
                "Target asset %s not available in current period — "
                "falling back to equal weight across %d assets",
                self._asset_symbol,
                len(symbols),
            )
            weights[:] = 1.0 / len(symbols)

        self.store_optimal_weights(weights)
        return weights
