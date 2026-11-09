"""Unit tests for HybridOptimizationStrategy.

Tests the two-phase hybrid approach: EA exploration + SLSQP polishing,
with lightweight settings for fast execution.

Optimization is expensive, so fixtures are cached at class scope to avoid
redundant ``optimize()`` calls — multiple assertions share a single run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from platypus import NSGAII

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.hybrid_optimization import (
    HybridOptimizationStrategy,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

SYMBOLS = ["A", "B", "C"]

_MU = pd.Series([0.01, 0.02, 0.015], index=SYMBOLS)
_COV = pd.DataFrame(
    [
        [0.04, 0.005, 0.002],
        [0.005, 0.09, 0.003],
        [0.002, 0.003, 0.01],
    ],
    index=SYMBOLS,
    columns=SYMBOLS,
)


def _make_hybrid_strategy(
    seed_weights: list[np.ndarray] | None = None,
    n_generations: int = 50,
    n_restarts: int = 1,
    population_size: int = 20,
    n_decomposition_directions: int = 5,
    include_markowitz: bool = False,
) -> HybridOptimizationStrategy:
    """Build a lightweight hybrid strategy for testing."""
    return HybridOptimizationStrategy(
        expected_returns=_MU,
        covariance_matrix=_COV,
        risk_aversion=0.5,
        algorithm=NSGAII,
        algorithm_kwargs={
            "variator": PortfolioVariator(min_assets=1, max_assets=3),
        },
        model=BalancedPortfoliosRiskContributions,
        model_kwargs={
            "l_assets": 1,
            "u_assets": 3,
            "constraint_slack": 0.1,
            "hhi_as_constraint": False,
            "hhi_bound": 0.5,
        },
        population_size=population_size,
        n_generations=n_generations,
        n_restarts=n_restarts,
        include_markowitz_front=include_markowitz,
        seed_weights=seed_weights,
        ea_budget_ratio=0.5,
        n_decomposition_directions=n_decomposition_directions,
        n_slsqp_random_starts=1,
        slsqp_maxiter=100,
    )


# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def hybrid_result() -> OptimizationResult:
    """Run the hybrid optimization once and share across tests."""
    strategy = _make_hybrid_strategy()
    return strategy.optimize()


@pytest.fixture(scope="module")
def hybrid_warm_start_result() -> OptimizationResult:
    """Run the hybrid optimization with warm-start seeds once."""
    seeds = [
        np.array([0.4, 0.3, 0.3]),
        np.array([0.5, 0.3, 0.2]),
    ]
    strategy = _make_hybrid_strategy(seed_weights=seeds)
    return strategy.optimize()


# ── Optimize Tests ───────────────────────────────────────────────────────────


@pytest.mark.slow
class TestHybridOptimize:
    """Test the full hybrid optimize() pipeline."""

    def test_returns_optimization_result(
        self, hybrid_result: OptimizationResult
    ) -> None:
        assert isinstance(hybrid_result, OptimizationResult)

    def test_front_is_non_empty(self, hybrid_result: OptimizationResult) -> None:
        assert len(hybrid_result.front) > 0

    def test_front_solutions_have_objectives(
        self, hybrid_result: OptimizationResult
    ) -> None:
        for sol in hybrid_result.front:
            assert sol.objectives is not None
            assert len(sol.objectives) >= 2

    def test_front_weights_sum_to_one(self, hybrid_result: OptimizationResult) -> None:
        for sol in hybrid_result.front:
            total = sum(sol.variables)
            assert abs(total - 1.0) < 1e-3, f"Weights sum={total}"

    def test_front_weights_non_negative(
        self, hybrid_result: OptimizationResult
    ) -> None:
        for sol in hybrid_result.front:
            assert all(w >= -1e-8 for w in sol.variables)

    def test_all_archives_from_both_phases(
        self, hybrid_result: OptimizationResult
    ) -> None:
        """Archives should contain results from EA + decomposition."""
        # EA archives + decomposition archives
        assert len(hybrid_result.all_archives) >= 2


# ── Merge Tests ──────────────────────────────────────────────────────────────


class TestHybridMergeFronts:
    """Test the _merge_fronts helper."""

    def test_empty_fronts_return_empty(self) -> None:
        merged = HybridOptimizationStrategy._merge_fronts([[], []])
        assert merged == []

    @pytest.mark.slow
    def test_merge_deduplicates(self, hybrid_result: OptimizationResult) -> None:
        """Identical solutions should be removed."""
        # Duplicate the front
        merged = HybridOptimizationStrategy._merge_fronts(
            [hybrid_result.front, hybrid_result.front]
        )
        assert len(merged) <= len(hybrid_result.front)


# ── Warm-Start Tests ────────────────────────────────────────────────────────


@pytest.mark.slow
class TestHybridWarmStart:
    """Test warm-start seeding in the hybrid strategy."""

    def test_warm_start_produces_valid_result(
        self, hybrid_warm_start_result: OptimizationResult
    ) -> None:
        assert len(hybrid_warm_start_result.front) > 0

    def test_warm_start_weights_are_valid(
        self, hybrid_warm_start_result: OptimizationResult
    ) -> None:
        for sol in hybrid_warm_start_result.front:
            total = sum(sol.variables)
            assert abs(total - 1.0) < 1e-3


# ── Selection Integration Tests ──────────────────────────────────────────────


@pytest.mark.slow
class TestHybridSelection:
    """Test that hybrid front works with selection strategies."""

    def test_sharpe_selection(self, hybrid_result: OptimizationResult) -> None:
        from investment_tool.strategies.selection_strategies import SharpeSelection

        weights = SharpeSelection().select(hybrid_result.front, pd.Index(SYMBOLS))
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-3
        assert (weights >= -1e-8).all()
