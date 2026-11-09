"""Unit tests for DecompositionOptimizationStrategy.

Tests reference direction generation, ideal/nadir estimation,
the full optimize() pipeline, and warm-start seeding — all with
lightweight settings for fast execution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from investment_tool.strategies.optimization_strategies.decomposition_optimization import (
    DecompositionOptimizationStrategy,
    _generate_reference_directions,
)
from investment_tool.strategies.optimization_strategies.optimization_result import (
    OptimizationResult,
)
from investment_tool.strategies.optimization_strategies.scalarized_solver import (
    ScalarizationMethod,
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
    n_directions: int = 10,
    seed_weights: list[np.ndarray] | None = None,
    hhi_as_constraint: bool = False,
    method: ScalarizationMethod = ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
) -> DecompositionOptimizationStrategy:
    """Build a lightweight decomposition strategy for testing."""
    return DecompositionOptimizationStrategy(
        expected_returns=_MU,
        covariance_matrix=_COV,
        l_assets=1,
        u_assets=3,
        n_directions=n_directions,
        n_random_starts=2,
        maxiter=100,
        hhi_as_constraint=hhi_as_constraint,
        scalarization_method=method,
        seed_weights=seed_weights,
    )


# ── Reference Direction Tests ────────────────────────────────────────────────


class TestReferenceDirections:
    """Test _generate_reference_directions utility."""

    def test_2d_directions_shape(self) -> None:
        ref = _generate_reference_directions(2, 10)
        assert ref.ndim == 2
        assert ref.shape[1] == 2
        assert ref.shape[0] >= 3  # At least a few directions

    def test_3d_directions_shape(self) -> None:
        ref = _generate_reference_directions(3, 20)
        assert ref.ndim == 2
        assert ref.shape[1] == 3

    def test_directions_sum_to_one(self) -> None:
        ref = _generate_reference_directions(3, 15)
        for row in ref:
            assert abs(row.sum() - 1.0) < 1e-6

    def test_directions_non_negative(self) -> None:
        ref = _generate_reference_directions(2, 10)
        assert (ref >= -1e-10).all()


# ── Optimize Tests ───────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDecompositionOptimize:
    """Test the full optimize() pipeline."""

    def test_returns_optimization_result(self) -> None:
        strategy = _make_strategy(n_directions=5)
        result = strategy.optimize()
        assert isinstance(result, OptimizationResult)

    def test_front_is_non_empty(self) -> None:
        strategy = _make_strategy(n_directions=5)
        result = strategy.optimize()
        assert len(result.front) > 0

    def test_front_solutions_have_objectives(self) -> None:
        strategy = _make_strategy(n_directions=5)
        result = strategy.optimize()
        for sol in result.front:
            assert sol.objectives is not None
            assert len(sol.objectives) == 3  # 3-obj mode

    def test_front_weights_sum_to_one(self) -> None:
        strategy = _make_strategy(n_directions=8)
        result = strategy.optimize()
        for sol in result.front:
            total = sum(sol.variables)
            assert abs(total - 1.0) < 1e-4, f"Weights sum={total}"

    def test_front_weights_non_negative(self) -> None:
        strategy = _make_strategy(n_directions=5)
        result = strategy.optimize()
        for sol in result.front:
            assert all(w >= -1e-8 for w in sol.variables)

    def test_2_objective_mode(self) -> None:
        strategy = _make_strategy(n_directions=5, hhi_as_constraint=True)
        result = strategy.optimize()
        assert len(result.front) > 0
        for sol in result.front:
            assert len(sol.objectives) == 2

    def test_all_archives_populated(self) -> None:
        strategy = _make_strategy(n_directions=5)
        result = strategy.optimize()
        assert len(result.all_archives) == 1
        assert len(result.all_archives[0]) > 0


# ── Warm-Start Tests ────────────────────────────────────────────────────────


@pytest.mark.slow
class TestDecompositionWarmStart:
    """Test warm-start seeding from prior-period solutions."""

    def test_warm_start_produces_valid_result(self) -> None:
        seeds = [
            np.array([0.4, 0.3, 0.3]),
            np.array([0.5, 0.3, 0.2]),
        ]
        strategy = _make_strategy(n_directions=5, seed_weights=seeds)
        result = strategy.optimize()
        assert len(result.front) > 0

    def test_warm_start_with_mismatched_seeds(self) -> None:
        """Seeds with wrong dimension should be handled gracefully."""
        seeds = [np.array([0.5, 0.5])]  # 2 assets, but problem has 3
        strategy = _make_strategy(n_directions=5, seed_weights=seeds)
        # Should still work — seed matching will fail but random starts remain
        result = strategy.optimize()
        assert len(result.front) > 0


# ── ScalarizationMethod Tests ───────────────────────────────────────────────


@pytest.mark.slow
class TestDecompositionScalarization:
    """Test different scalarization methods produce valid fronts."""

    def test_weighted_sum_method(self) -> None:
        strategy = _make_strategy(
            n_directions=5,
            method=ScalarizationMethod.WEIGHTED_SUM,
        )
        result = strategy.optimize()
        assert len(result.front) > 0

    def test_tchebycheff_method(self) -> None:
        strategy = _make_strategy(
            n_directions=5,
            method=ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
        )
        result = strategy.optimize()
        assert len(result.front) > 0
