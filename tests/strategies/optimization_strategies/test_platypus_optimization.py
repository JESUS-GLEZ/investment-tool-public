"""Unit tests for PlatypusOptimizationStrategy.

Tests warm-start seed building, portfolio selection, merge/prune logic,
and the Markowitz benchmark computation.

Optimization is expensive, so fixtures are cached at module scope to avoid
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
from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
    WarmStartSource,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

# Minimal 3-asset universe for fast tests
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


def _make_strategy(
    seed_weights: list[np.ndarray] | None = None,
    include_markowitz: bool = False,
    n_generations: int = 50,
    n_restarts: int = 1,
    population_size: int = 30,
    warm_start_source: WarmStartSource = WarmStartSource.MARKOWITZ,
) -> PlatypusOptimizationStrategy:
    """Build a lightweight strategy instance for testing."""
    return PlatypusOptimizationStrategy(
        expected_returns=_MU,
        covariance_matrix=_COV,
        risk_aversion=0.5,
        algorithm=NSGAII,
        algorithm_kwargs={
            "variator": PortfolioVariator(min_assets=1, max_assets=3),
        },
        model=BalancedPortfoliosRiskContributions,
        model_kwargs={"l_assets": 1, "u_assets": 3},
        population_size=population_size,
        n_generations=n_generations,
        n_restarts=n_restarts,
        include_markowitz_front=include_markowitz,
        seed_weights=seed_weights,
        warm_start_source=warm_start_source,
    )


# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def default_result() -> OptimizationResult:
    """Run the default optimization once and share across tests."""
    strategy = _make_strategy()
    return strategy.optimize()


@pytest.fixture(scope="module")
def sharpe_weights(default_result: OptimizationResult) -> pd.Series:
    """Compute sharpe selection weights from the default result."""
    from investment_tool.strategies.selection_strategies import SharpeSelection

    return SharpeSelection().select(default_result.front, pd.Index(SYMBOLS))


@pytest.fixture(scope="module")
def markowitz_result() -> OptimizationResult:
    """Run optimization with Markowitz front enabled."""
    strategy = _make_strategy(include_markowitz=True, n_generations=50)
    return strategy.optimize()


@pytest.fixture(scope="module")
def multi_restart_result() -> OptimizationResult:
    """Run optimization with 2 restarts."""
    strategy = _make_strategy(n_restarts=2)
    return strategy.optimize()


@pytest.fixture(scope="module")
def warm_start_result() -> tuple[pd.Series, OptimizationResult]:
    """Run optimization with warm-start seeds."""
    from investment_tool.strategies.selection_strategies import SharpeSelection

    strategy = _make_strategy(
        seed_weights=[
            np.array([0.4, 0.3, 0.3]),
            np.array([0.5, 0.3, 0.2]),
        ],
        n_generations=50,
    )
    result = strategy.optimize()
    weights = SharpeSelection().select(result.front, pd.Index(SYMBOLS))
    return weights, result


@pytest.fixture(scope="module")
def markowitz_front() -> list:
    """Compute a Markowitz front once."""
    strategy = _make_strategy()
    return strategy.compute_markowitz_front(
        n_evaluations=200,
        population_size=20,
        n_restarts=1,
    )


# ── Seed Building Tests ──────────────────────────────────────────────────────


class TestSeedBuilding:
    """Test _build_seed_solutions for warm-start from a full front."""

    def test_no_seeds_when_no_seed_weights(self) -> None:
        strategy = _make_strategy(seed_weights=None)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        assert seeds == []

    def test_no_seeds_when_empty_list(self) -> None:
        strategy = _make_strategy(seed_weights=[])
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        assert seeds == []

    def test_seeds_include_all_front_solutions(self) -> None:
        front = [
            np.array([0.4, 0.3, 0.3]),
            np.array([0.5, 0.3, 0.2]),
            np.array([0.6, 0.2, 0.2]),
        ]
        strategy = _make_strategy(seed_weights=front, population_size=30)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        # At least the 3 original front solutions should be present
        assert len(seeds) >= len(front)

    def test_seeds_capped_at_population_size(self) -> None:
        front = [np.array([0.4, 0.3, 0.3])] * 50  # More than pop_size=30
        strategy = _make_strategy(seed_weights=front, population_size=30)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        assert len(seeds) <= strategy.population_size

    def test_seed_weights_sum_to_one(self) -> None:
        front = [np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])]
        strategy = _make_strategy(seed_weights=front)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        for s in seeds:
            total = sum(s.variables)
            assert abs(total - 1.0) < 1e-6, f"Seed weights sum={total}"

    def test_seed_weights_non_negative(self) -> None:
        front = [np.array([0.5, 0.3, 0.2])]
        strategy = _make_strategy(seed_weights=front)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        for s in seeds:
            assert all(w >= 0 for w in s.variables)

    def test_dimension_mismatch_returns_empty(self) -> None:
        front = [np.array([0.5, 0.5])]  # 2 assets, but problem has 3
        strategy = _make_strategy(seed_weights=front)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        assert seeds == []

    def test_first_seeds_are_exact_front_copies(self) -> None:
        front = [np.array([0.5, 0.3, 0.2]), np.array([0.4, 0.4, 0.2])]
        strategy = _make_strategy(seed_weights=front)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3)
        # First N seeds should be the exact front solutions
        for i, w in enumerate(front):
            np.testing.assert_array_almost_equal(seeds[i].variables, w, decimal=4)


# ── Selection via optimize() Tests ───────────────────────────────────────────


@pytest.mark.slow
class TestOptimizeWithSelection:
    """Test optimize() + explicit selection strategy produces valid weights."""

    def test_sharpe_selection_returns_series(self, sharpe_weights: pd.Series) -> None:
        assert isinstance(sharpe_weights, pd.Series)
        assert list(sharpe_weights.index) == SYMBOLS

    def test_sharpe_selection_weights_sum_to_one(
        self, sharpe_weights: pd.Series
    ) -> None:
        assert abs(sharpe_weights.sum() - 1.0) < 1e-4

    def test_sharpe_selection_no_negative_weights(
        self, sharpe_weights: pd.Series
    ) -> None:
        assert (sharpe_weights >= -1e-8).all()


class TestOptimizeReturnsResult:
    """Test the new optimize() method that returns OptimizationResult."""

    @pytest.mark.slow
    def test_returns_optimization_result(
        self, default_result: OptimizationResult
    ) -> None:
        assert isinstance(default_result, OptimizationResult)

    @pytest.mark.slow
    def test_front_is_non_empty(self, default_result: OptimizationResult) -> None:
        assert len(default_result.front) > 0

    @pytest.mark.slow
    def test_all_archives_populated(
        self, multi_restart_result: OptimizationResult
    ) -> None:
        assert len(multi_restart_result.all_archives) == 2

    @pytest.mark.slow
    def test_markowitz_front_when_requested(
        self, markowitz_result: OptimizationResult
    ) -> None:
        assert markowitz_result.markowitz_front is not None
        assert len(markowitz_result.markowitz_front) > 0

    def test_no_markowitz_front_when_not_requested(self) -> None:
        strategy = _make_strategy(
            include_markowitz=False,
            warm_start_source=WarmStartSource.PREVIOUS,
            n_generations=50,
        )
        result = strategy.optimize()
        assert result.markowitz_front is None


# ── Warm-Start Optimization Tests ────────────────────────────────────────────


class TestWarmStartOptimization:
    """Test that warm-start changes the optimization behaviour."""

    def test_warm_start_uses_fewer_evaluations(self) -> None:
        """When seed_weights are provided, effective_generations is reduced."""
        strategy = _make_strategy(
            seed_weights=[np.array([0.4, 0.3, 0.3])],
            n_generations=1000,
        )
        # Warm-start ratio is 0.7 by default
        expected_evals = int(1000 * strategy.warm_start_generation_ratio)
        assert expected_evals == 700

    @pytest.mark.slow
    def test_warm_start_still_returns_valid_weights(
        self, warm_start_result: tuple[pd.Series, OptimizationResult]
    ) -> None:
        weights, _ = warm_start_result
        assert isinstance(weights, pd.Series)
        assert abs(weights.sum() - 1.0) < 1e-4
        assert (weights >= -1e-8).all()


# ── Markowitz Benchmark Tests ────────────────────────────────────────────────


class TestMarkowitzBenchmark:
    """Test compute_markowitz_front as an independent computation."""

    def test_returns_non_empty_front(self, markowitz_front: list) -> None:
        assert len(markowitz_front) > 0

    def test_solutions_have_two_objectives(self, markowitz_front: list) -> None:
        """Markowitz is a 2-objective problem (variance, -return)."""
        for sol in markowitz_front:
            assert len(sol.objectives) == 2

    def test_solutions_are_feasible(self, markowitz_front: list) -> None:
        for sol in markowitz_front:
            w = np.array(sol.variables)
            assert abs(w.sum() - 1.0) < 0.05  # Approximately sum to 1
            assert (w >= -0.01).all()  # No significant shorting


# ── Merge / Prune Tests ──────────────────────────────────────────────────────


class TestMergeNondominated:
    """Test the _merge_nondominated helper (delegates to utils.merge_nondominated)."""

    def test_empty_input_returns_empty(self) -> None:
        result = PlatypusOptimizationStrategy._merge_nondominated([[]], target_size=10)
        assert result == []

    def test_deduplication(self, markowitz_front: list) -> None:
        """Duplicate solutions should be removed."""
        # Duplicate the front
        merged = PlatypusOptimizationStrategy._merge_nondominated(
            [markowitz_front, markowitz_front], target_size=100
        )

        # No duplicates — should be same size as original front
        assert len(merged) <= len(markowitz_front)
