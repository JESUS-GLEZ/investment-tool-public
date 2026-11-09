"""Named configurations of existing investment strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from investment_tool.constants import (
    StrategyType,
    OptimizationReturnEstimator,
    PortfolioBalance,
)
from investment_tool.defaults import STRATEGY_OPTIMIZER
from investment_tool.strategies.selection_strategies import (
    BaseSelectionStrategy,
    ReturnTargetSelection,
    ReturnWithRiskBudgetSelection,
    SharpeSelection,
)

SelectionMethod = Literal["sharpe", "return_with_risk_budget", "return_target"]


def build_selector(
    method: SelectionMethod,
    volatility_increase: float = 0.25,
    return_target_fraction: float = 0.75,
) -> BaseSelectionStrategy:
    if method == "sharpe":
        return SharpeSelection()
    if method == "return_with_risk_budget":
        return ReturnWithRiskBudgetSelection(volatility_increase)
    if method == "return_target":
        return ReturnTargetSelection(return_target_fraction)
    raise ValueError(f"Unknown selection method: {method}")


@dataclass
class StrategyVariant:
    """Overrides, not another strategy implementation. None inherits run defaults."""

    name: str
    strategy_type: StrategyType = StrategyType.OPTIMIZATION
    return_estimator: OptimizationReturnEstimator | None = None
    balance: PortfolioBalance = PortfolioBalance.RISK_CONTRIBUTIONS
    news_shortlist: bool = False
    news_top_assets: int = 50
    news_min_articles: int = 3
    selection: SelectionMethod = "sharpe"
    return_target_fraction: float = 0.75
    volatility_increase: float = 0.25
    population_size: int | None = None
    n_generations: int | None = None
    min_assets: int | None = None
    max_assets: int | None = None
    n_restarts: int = STRATEGY_OPTIMIZER.n_restarts
    random_seed: int = STRATEGY_OPTIMIZER.random_seed
    search_seconds: float | None = STRATEGY_OPTIMIZER.search_seconds
    seed_max_subsets: int = STRATEGY_OPTIMIZER.seed_max_subsets
    seed_beam_width: int = STRATEGY_OPTIMIZER.seed_beam_width
    refinement_supports: int = STRATEGY_OPTIMIZER.refinement_supports

    asset_symbol: str | None = None
    top_n: int | None = None
    lookback: int | None = None
    vol_window: int | None = None
    trend_window: int | None = None
    min_weight: float | None = None

    @property
    def uses_optimizer(self) -> bool:
        return self.strategy_type in (
            StrategyType.OPTIMIZATION,
            StrategyType.COMBINED,
            StrategyType.SENTIMENT_EWM,
        )

    def constructor_kwargs(
        self, selector: BaseSelectionStrategy | None = None
    ) -> dict[str, Any]:
        """Configuration only: both execution paths use the existing strategy factory."""
        if self.strategy_type == StrategyType.BUY_AND_HOLD:
            return {"asset_symbol": self.asset_symbol or "SPY"}
        if self.strategy_type == StrategyType.MOMENTUM_RISK_PARITY:
            return {
                key: value
                for key, value in (
                    (key, getattr(self, key))
                    for key in (
                        "top_n",
                        "lookback",
                        "vol_window",
                        "trend_window",
                        "min_weight",
                    )
                )
                if value is not None
            }
        kwargs: dict[str, Any] = {
            key: value
            for key in ("population_size", "n_generations", "n_restarts")
            if (value := getattr(self, key)) is not None
        }
        kwargs["selection_strategy"] = selector or build_selector(
            self.selection, self.volatility_increase, self.return_target_fraction
        )
        if self.strategy_type == StrategyType.OPTIMIZATION:
            kwargs.update(
                {
                    key: getattr(self, key)
                    for key in (
                        "balance",
                        "news_shortlist",
                        "news_top_assets",
                        "news_min_articles",
                    )
                }
            )
        if self.return_estimator is not None:
            kwargs["return_estimator"] = self.return_estimator
        return kwargs

    def search_settings(self) -> dict[str, Any]:
        return (
            {
                key: getattr(self, key)
                for key in (
                    "random_seed",
                    "search_seconds",
                    "seed_max_subsets",
                    "seed_beam_width",
                    "refinement_supports",
                )
            }
            if self.uses_optimizer
            else {}
        )
