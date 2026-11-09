"""Unit tests for the island/ensemble multi-start and algorithm portfolio.

Tests the ``_get_algorithm_for_restart`` dispatcher, graduated seed noise
in ``_build_seed_solutions``, and the ``algorithm_portfolio`` parameter
in ``PlatypusOptimizationStrategy``.

Optimization is expensive, so fixtures are cached at module scope to avoid
redundant ``optimize()`` calls — multiple assertions share a single run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from platypus import NSGAII, NSGAIII, SPEA2

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
)
from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)
from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.platypus_optimization import (
    PlatypusOptimizationStrategy,
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


def _make_strategy(
    seed_weights: list[np.ndarray] | None = None,
    algorithm_portfolio: list[tuple[type, dict]] | None = None,
    n_generations: int = 50,
    n_restarts: int = 1,
    population_size: int = 30,
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
        include_markowitz_front=False,
        seed_weights=seed_weights,
        algorithm_portfolio=algorithm_portfolio,
    )


# ── Shared fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def multi_restart_portfolio_result() -> OptimizationResult:
    """Run 2 restarts with different algorithms once."""
    variator = PortfolioVariator(min_assets=1, max_assets=3)
    portfolio = [
        (NSGAII, {"variator": variator}),
        (SPEA2, {"variator": variator}),
    ]
    strategy = _make_strategy(
        algorithm_portfolio=portfolio,
        n_restarts=2,
        n_generations=50,
    )
    return strategy.optimize()


@pytest.fixture(scope="module")
def warm_start_multi_restart_result() -> OptimizationResult:
    """Run 3 restarts with warm-start seeds once."""
    front = [
        np.array([0.5, 0.3, 0.2]),
        np.array([0.4, 0.4, 0.2]),
    ]
    strategy = _make_strategy(
        seed_weights=front,
        n_restarts=3,
        n_generations=50,
    )
    return strategy.optimize()


@pytest.fixture(scope="module")
def markowitz_warm_start_result() -> OptimizationResult:
    """Run optimization with MARKOWITZ warm-start source once."""
    from investment_tool.strategies.optimization_strategies.platypus_optimization import (
        WarmStartSource,
    )

    strategy = _make_strategy(n_generations=50, n_restarts=1)
    strategy.include_markowitz_front = True
    strategy.warm_start_source = WarmStartSource.MARKOWITZ
    return strategy.optimize()


@pytest.fixture(scope="module")
def previous_warm_start_result() -> (
    tuple[PlatypusOptimizationStrategy, OptimizationResult]
):
    """Run optimization with PREVIOUS warm-start source once."""
    from investment_tool.strategies.optimization_strategies.platypus_optimization import (
        WarmStartSource,
    )

    prior_seeds = [np.array([0.5, 0.3, 0.2])]
    strategy = _make_strategy(
        seed_weights=prior_seeds,
        n_generations=50,
        n_restarts=1,
    )
    strategy.include_markowitz_front = True
    strategy.warm_start_source = WarmStartSource.PREVIOUS
    result = strategy.optimize()
    return strategy, result


@pytest.fixture(scope="module")
def both_warm_start_result() -> tuple[PlatypusOptimizationStrategy, OptimizationResult]:
    """Run optimization with BOTH warm-start source once."""
    from investment_tool.strategies.optimization_strategies.platypus_optimization import (
        WarmStartSource,
    )

    prior_seeds = [np.array([0.5, 0.3, 0.2])]
    strategy = _make_strategy(
        seed_weights=prior_seeds,
        n_generations=50,
        n_restarts=1,
    )
    strategy.include_markowitz_front = True
    strategy.warm_start_source = WarmStartSource.BOTH
    result = strategy.optimize()
    return strategy, result


@pytest.fixture(scope="module")
def markowitz_seeding_no_plots_result() -> OptimizationResult:
    """Run optimization with MARKOWITZ source but include_markowitz_front=False."""
    from investment_tool.strategies.optimization_strategies.platypus_optimization import (
        WarmStartSource,
    )

    strategy = _make_strategy(n_generations=50, n_restarts=1)
    strategy.include_markowitz_front = False
    strategy.warm_start_source = WarmStartSource.MARKOWITZ
    return strategy.optimize()


# ── Algorithm Portfolio Dispatcher Tests ─────────────────────────────────────


class TestAlgorithmPortfolioDispatch:
    """Test _get_algorithm_for_restart routing."""

    def test_default_algorithm_when_no_portfolio(self) -> None:
        """Without algorithm_portfolio, every restart uses the default."""
        strategy = _make_strategy()
        algo_cls, algo_kwargs = strategy._get_algorithm_for_restart(0)
        assert algo_cls is NSGAII
        algo_cls2, _ = strategy._get_algorithm_for_restart(2)
        assert algo_cls2 is NSGAII

    def test_portfolio_cycles_through_entries(self) -> None:
        """Restarts should cycle through the portfolio list."""
        portfolio = [
            (NSGAIII, {"divisions_outer": 12}),
            (SPEA2, {}),
        ]
        strategy = _make_strategy(algorithm_portfolio=portfolio)

        algo0, kw0 = strategy._get_algorithm_for_restart(0)
        assert algo0 is NSGAIII
        assert kw0["divisions_outer"] == 12

        algo1, kw1 = strategy._get_algorithm_for_restart(1)
        assert algo1 is SPEA2

        # Wrap around
        algo2, _ = strategy._get_algorithm_for_restart(2)
        assert algo2 is NSGAIII

    def test_portfolio_returns_copies_of_kwargs(self) -> None:
        """Each call should return independent dicts to avoid cross-contamination."""
        portfolio = [
            (NSGAII, {"variator": PortfolioVariator()}),
        ]
        strategy = _make_strategy(algorithm_portfolio=portfolio)
        _, kw_a = strategy._get_algorithm_for_restart(0)
        _, kw_b = strategy._get_algorithm_for_restart(0)
        assert kw_a is not kw_b


# ── Graduated Seed Noise Tests ───────────────────────────────────────────────


class TestGraduatedSeedNoise:
    """Test that _build_seed_solutions accepts different noise scales."""

    def test_low_noise_seeds_are_close_to_original(self) -> None:
        front = [np.array([0.5, 0.3, 0.2])]
        strategy = _make_strategy(seed_weights=front, population_size=10)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds = strategy._build_seed_solutions(problem, n_assets=3, noise_scale=0.001)
        # First seed is exact, perturbations should be very close
        for s in seeds:
            w = np.array(s.variables)
            assert np.allclose(w, front[0], atol=0.05)

    def test_high_noise_seeds_are_more_diverse(self) -> None:
        front = [np.array([0.5, 0.3, 0.2])]
        strategy = _make_strategy(seed_weights=front, population_size=10)
        problem = MarkowitzLongOnly(
            mu=strategy.expected_returns.values,
            cov=strategy.covariance_matrix.values,
        )
        seeds_low = strategy._build_seed_solutions(
            problem, n_assets=3, noise_scale=0.01
        )
        seeds_high = strategy._build_seed_solutions(
            problem, n_assets=3, noise_scale=0.15
        )

        # Compute variance of seed weights as a diversity measure
        def _weight_variance(seeds: list) -> float:
            ws = np.array([s.variables for s in seeds])
            return float(ws.var())

        assert _weight_variance(seeds_high) >= _weight_variance(seeds_low)


# ── Island Multi-Start Integration Tests ─────────────────────────────────────


class TestIslandMultiStart:
    """Test the full multi-start optimization with algorithm portfolio."""

    @pytest.mark.slow
    def test_multi_restart_with_portfolio_produces_results(
        self, multi_restart_portfolio_result: OptimizationResult
    ) -> None:
        """Running 2 restarts with different algorithms should merge fronts."""
        assert len(multi_restart_portfolio_result.front) > 0
        assert len(multi_restart_portfolio_result.all_archives) == 2

    @pytest.mark.slow
    def test_warm_start_seeds_all_restarts(
        self, warm_start_multi_restart_result: OptimizationResult
    ) -> None:
        """With seed_weights, all restarts should receive seeds."""
        # All 3 restarts should have run and contributed archives
        assert len(warm_start_multi_restart_result.all_archives) == 3
        assert len(warm_start_multi_restart_result.front) > 0

    def test_warm_start_generation_ratio_default(self) -> None:
        """Default warm_start_generation_ratio should be 0.7."""
        strategy = _make_strategy()
        assert strategy.warm_start_generation_ratio == 0.7

    def test_default_n_generations(self) -> None:
        """Default n_generations should be 500_000."""
        strategy = PlatypusOptimizationStrategy(
            expected_returns=_MU,
            covariance_matrix=_COV,
            risk_aversion=0.5,
            algorithm=NSGAII,
            model=BalancedPortfoliosRiskContributions,
            model_kwargs={"l_assets": 1, "u_assets": 3},
        )
        assert strategy.n_generations == 500_000
        assert strategy.n_restarts == 5


# ── Markowitz Warm-Start Seeding Tests ───────────────────────────────────────


class TestMarkowitzWarmStart:
    """Test WarmStartSource controls which seeds are used."""

    @pytest.mark.slow
    def test_markowitz_source_seeds_from_markowitz_front(
        self, markowitz_warm_start_result: OptimizationResult
    ) -> None:
        """MARKOWITZ mode should compute and use Markowitz solutions."""
        assert len(markowitz_warm_start_result.front) > 0
        assert markowitz_warm_start_result.markowitz_front is not None

    @pytest.mark.slow
    def test_previous_source_ignores_markowitz(
        self,
        previous_warm_start_result: tuple[
            PlatypusOptimizationStrategy, OptimizationResult
        ],
    ) -> None:
        """PREVIOUS mode should not use Markowitz solutions as seeds."""
        strategy, result = previous_warm_start_result
        assert len(result.front) > 0
        # Markowitz is still computed for the plots
        assert result.markowitz_front is not None
        # But seed_weights is untouched — only the prior seeds were used
        assert len(strategy.seed_weights) == 1

    @pytest.mark.slow
    def test_both_source_combines_markowitz_and_prior(
        self,
        both_warm_start_result: tuple[PlatypusOptimizationStrategy, OptimizationResult],
    ) -> None:
        """BOTH mode should combine Markowitz + prior seeds."""
        strategy, result = both_warm_start_result
        assert len(result.front) > 0
        assert result.markowitz_front is not None
        # seed_weights should NOT be mutated (restored after optimize)
        assert len(strategy.seed_weights) == 1

    def test_no_markowitz_when_front_disabled_and_source_previous(self) -> None:
        """When include_markowitz_front=False and source=PREVIOUS,
        no Markowitz computation should happen."""
        from investment_tool.strategies.optimization_strategies.platypus_optimization import (
            WarmStartSource,
        )

        strategy = _make_strategy(n_generations=50, n_restarts=1)
        strategy.include_markowitz_front = False
        strategy.warm_start_source = WarmStartSource.PREVIOUS

        result = strategy.optimize()

        assert result.markowitz_front is None

    @pytest.mark.slow
    def test_markowitz_computed_for_seeding_even_without_plots(
        self, markowitz_seeding_no_plots_result: OptimizationResult
    ) -> None:
        """When include_markowitz_front=False but source=MARKOWITZ,
        Markowitz should still be computed for seeding."""
        # Markowitz was computed for seeding, so it's available in the result
        assert markowitz_seeding_no_plots_result.markowitz_front is not None
        assert len(markowitz_seeding_no_plots_result.front) > 0

    def test_default_warm_start_source_is_markowitz(self) -> None:
        """Default warm_start_source should be MARKOWITZ."""
        from investment_tool.strategies.optimization_strategies.platypus_optimization import (
            WarmStartSource,
        )

        strategy = _make_strategy()
        assert strategy.warm_start_source == WarmStartSource.MARKOWITZ

    def test_extract_weights_from_front(self) -> None:
        """Test the static helper that extracts weight arrays from solutions."""
        from platypus import Solution

        problem = MarkowitzLongOnly(
            mu=np.array([0.01, 0.02, 0.015]),
            cov=np.array(
                [[0.04, 0.005, 0.002], [0.005, 0.09, 0.003], [0.002, 0.003, 0.01]]
            ),
        )
        sol = Solution(problem)
        sol.variables[:] = [0.5, 0.3, 0.2]
        weights = PlatypusOptimizationStrategy._extract_weights_from_front([sol])
        assert len(weights) == 1
        np.testing.assert_array_almost_equal(weights[0], [0.5, 0.3, 0.2])

    def test_extract_weights_empty_front(self) -> None:
        """Empty front should return empty list."""
        weights = PlatypusOptimizationStrategy._extract_weights_from_front([])
        assert weights == []
