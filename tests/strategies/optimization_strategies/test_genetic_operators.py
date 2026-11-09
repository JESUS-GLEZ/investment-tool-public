"""Unit tests for the PortfolioVariator genetic operators.

Tests the enhanced two-stage mutation (weight perturbation +
cardinality change), crossover logic, and simplex repair.
"""

from __future__ import annotations

import numpy as np
from platypus import Problem, Real, Solution

from investment_tool.strategies.optimization_strategies.custom_algorithm.genetic_operators import (
    PortfolioVariator,
    repair_portfolio,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _dummy_problem(n_assets: int = 5) -> Problem:
    """Create a minimal Platypus Problem for testing."""
    p = Problem(n_assets, 2)
    for i in range(n_assets):
        p.types[i] = Real(0.0, 1.0)
    return p


def _make_solution(
    problem: Problem,
    weights: np.ndarray,
) -> Solution:
    """Wrap a weight vector in a Platypus Solution."""
    s = Solution(problem)
    s.variables[:] = weights.tolist()
    return s


# ── repair_portfolio Tests ───────────────────────────────────────────────────


class TestRepairPortfolio:
    """Test the simplex projection utility."""

    def test_normal_weights_normalized(self) -> None:
        x = np.array([0.3, 0.3, 0.4])
        result = repair_portfolio(x.copy())
        assert abs(result.sum() - 1.0) < 1e-10

    def test_dust_positions_zeroed(self) -> None:
        x = np.array([0.5, 0.0005, 0.4995])
        result = repair_portfolio(x.copy(), dust_tol=1e-3)
        assert result[1] == 0.0
        assert abs(result.sum() - 1.0) < 1e-10

    def test_all_zeros_gives_uniform(self) -> None:
        x = np.zeros(4)
        result = repair_portfolio(x.copy())
        np.testing.assert_array_almost_equal(result, [0.25, 0.25, 0.25, 0.25])

    def test_result_non_negative(self) -> None:
        x = np.array([0.6, -0.1, 0.5])
        result = repair_portfolio(np.clip(x, 0, None))
        assert (result >= 0.0).all()

    def test_preserves_active_set(self) -> None:
        x = np.array([0.5, 0.0, 0.5, 0.0])
        result = repair_portfolio(x.copy())
        assert result[1] == 0.0
        assert result[3] == 0.0
        assert result[0] > 0.0
        assert result[2] > 0.0


# ── PortfolioVariator Construction Tests ─────────────────────────────────────


class TestVariatorConstruction:
    """Test PortfolioVariator construction and defaults."""

    def test_defaults(self) -> None:
        v = PortfolioVariator()
        assert v.min_assets == 1
        assert v.max_assets is None
        assert v.mutation_prob == 0.5
        assert v.perturb_prob == 0.7
        assert v.cardinality_prob == 0.15
        assert v.noise_std == 0.05
        assert v.arity == 2

    def test_custom_params(self) -> None:
        v = PortfolioVariator(
            min_assets=2,
            max_assets=10,
            mutation_prob=0.8,
            perturb_prob=0.5,
            cardinality_prob=0.3,
            noise_std=0.1,
        )
        assert v.min_assets == 2
        assert v.max_assets == 10
        assert v.mutation_prob == 0.8
        assert v.perturb_prob == 0.5
        assert v.cardinality_prob == 0.3
        assert v.noise_std == 0.1

    def test_min_assets_floor(self) -> None:
        """min_assets should be at least 1."""
        v = PortfolioVariator(min_assets=0)
        assert v.min_assets == 1


# ── Crossover Tests ──────────────────────────────────────────────────────────


class TestCrossover:
    """Test the union-average crossover."""

    def test_children_non_negative(self) -> None:
        v = PortfolioVariator()
        p1 = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        p2 = np.array([0.0, 0.0, 0.3, 0.3, 0.4])
        c1, c2 = v._crossover(p1, p2)
        assert (c1 >= 0.0).all()
        assert (c2 >= 0.0).all()

    def test_common_assets_averaged(self) -> None:
        v = PortfolioVariator()
        p1 = np.array([0.0, 0.6, 0.4])
        p2 = np.array([0.0, 0.4, 0.6])
        c1, c2 = v._crossover(p1, p2)
        # Assets 1 and 2 are common — children should get the average
        np.testing.assert_almost_equal(c1[1], 0.5)
        np.testing.assert_almost_equal(c1[2], 0.5)

    def test_unique_assets_inherited(self) -> None:
        v = PortfolioVariator()
        p1 = np.array([0.5, 0.5, 0.0])
        p2 = np.array([0.0, 0.5, 0.5])
        c1, c2 = v._crossover(p1, p2)
        np.testing.assert_allclose(c1, [0.5, 0.5, 0.0])
        np.testing.assert_allclose(c2, [0.0, 0.5, 0.5])
        swapped1, swapped2 = v._crossover(p2, p1)
        np.testing.assert_array_equal(c1, swapped2)
        np.testing.assert_array_equal(c2, swapped1)


# ── Perturbation Mutation Tests ──────────────────────────────────────────────


class TestPerturbWeights:
    """Test the gentle weight-perturbation mutation."""

    def test_perturbation_preserves_active_set(self) -> None:
        v = PortfolioVariator(noise_std=0.01)
        x = np.array([0.5, 0.0, 0.3, 0.2, 0.0])
        result = v._perturb_weights(x.copy())
        # Zero positions should stay zero
        assert result[1] == 0.0
        assert result[4] == 0.0

    def test_perturbation_changes_weights(self) -> None:
        """With noise_std > 0, perturbation should usually change weights."""
        np.random.seed(42)
        v = PortfolioVariator(noise_std=0.1)
        x = np.array([0.5, 0.3, 0.2])
        changed = False
        for _ in range(10):
            result = v._perturb_weights(x.copy())
            if not np.allclose(result, x):
                changed = True
                break
        assert changed

    def test_perturbation_result_non_negative(self) -> None:
        np.random.seed(123)
        v = PortfolioVariator(noise_std=0.05)
        x = np.array([0.5, 0.3, 0.2])
        for _ in range(20):
            result = v._perturb_weights(x.copy())
            assert (result >= 0.0).all()

    def test_empty_active_set_returns_unchanged(self) -> None:
        v = PortfolioVariator()
        x = np.zeros(3)
        result = v._perturb_weights(x.copy())
        np.testing.assert_array_equal(result, x)


# ── Cardinality Mutation Tests ───────────────────────────────────────────────


class TestCardinalityMutation:
    """Test the structural cardinality-change mutation."""

    def test_can_add_positions(self) -> None:
        np.random.seed(42)
        v = PortfolioVariator(min_assets=1, max_assets=5)
        x = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        added = False
        for _ in range(30):
            result = v._mutate_cardinality(x.copy())
            if np.count_nonzero(result > 0) > 2:
                added = True
                break
        assert added

    def test_can_drop_positions(self) -> None:
        np.random.seed(42)
        v = PortfolioVariator(min_assets=1, max_assets=5)
        x = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        dropped = False
        for _ in range(30):
            result = v._mutate_cardinality(x.copy())
            if np.count_nonzero(result > 0) < 5:
                dropped = True
                break
        assert dropped

    def test_respects_min_assets(self) -> None:
        np.random.seed(42)
        v = PortfolioVariator(min_assets=2, max_assets=5)
        x = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        for _ in range(20):
            result = v._mutate_cardinality(x.copy())
            assert np.count_nonzero(result > 0) >= 2


# ── Two-Stage Mutation Dispatch Tests ────────────────────────────────────────


class TestMutateDispatch:
    """Test the two-stage _mutate dispatcher."""

    def test_no_mutation_when_probability_zero(self) -> None:
        v = PortfolioVariator(mutation_prob=0.0)
        x = np.array([0.5, 0.3, 0.2])
        result = v._mutate(x.copy())
        np.testing.assert_array_equal(result, x)

    def test_always_mutates_when_probability_one(self) -> None:
        """With mutation_prob=1.0, the output should differ from input (usually)."""
        np.random.seed(42)
        v = PortfolioVariator(mutation_prob=1.0, noise_std=0.1)
        x = np.array([0.5, 0.3, 0.2])
        changed = False
        for _ in range(10):
            result = v._mutate(x.copy())
            if not np.allclose(result, x):
                changed = True
                break
        assert changed


# ── Evolve (End-to-End) Tests ────────────────────────────────────────────────


class TestEvolve:
    """Test the full evolve() method producing valid offspring."""

    def test_produces_two_children(self) -> None:
        problem = _dummy_problem(3)
        v = PortfolioVariator(min_assets=1, max_assets=3)
        p1 = _make_solution(problem, np.array([0.5, 0.3, 0.2]))
        p2 = _make_solution(problem, np.array([0.4, 0.4, 0.2]))
        children = v.evolve([p1, p2])
        assert len(children) == 2

    def test_children_weights_sum_to_one(self) -> None:
        problem = _dummy_problem(5)
        v = PortfolioVariator(min_assets=1, max_assets=5, mutation_prob=1.0)
        p1 = _make_solution(problem, np.array([0.3, 0.3, 0.2, 0.1, 0.1]))
        p2 = _make_solution(problem, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
        for _ in range(10):
            children = v.evolve([p1, p2])
            for child in children:
                total = sum(child.variables)
                assert abs(total - 1.0) < 1e-6

    def test_children_weights_non_negative(self) -> None:
        problem = _dummy_problem(4)
        v = PortfolioVariator(min_assets=1, max_assets=4, mutation_prob=1.0)
        p1 = _make_solution(problem, np.array([0.4, 0.3, 0.2, 0.1]))
        p2 = _make_solution(problem, np.array([0.25, 0.25, 0.25, 0.25]))
        for _ in range(10):
            children = v.evolve([p1, p2])
            for child in children:
                assert all(w >= 0.0 for w in child.variables)

    def test_children_are_solution_objects(self) -> None:
        problem = _dummy_problem(3)
        v = PortfolioVariator(min_assets=1, max_assets=3)
        p1 = _make_solution(problem, np.array([0.5, 0.3, 0.2]))
        p2 = _make_solution(problem, np.array([0.6, 0.2, 0.2]))
        children = v.evolve([p1, p2])
        for child in children:
            assert isinstance(child, Solution)
            assert child.problem is problem
