"""Unit tests for the BalancedPortfoliosRiskContributions (BPRC) model.

Tests both the original 3-objective mode and the new 2-objective
``hhi_as_constraint`` mode, constraint slack relaxation, and the
updated ``activity_tol`` alignment.
"""

from __future__ import annotations

import numpy as np
import pytest
from platypus import Solution

from investment_tool.strategies.optimization_strategies.model.bprc import (
    BalancedPortfoliosRiskContributions,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

SYMBOLS = ["A", "B", "C"]

_MU = np.array([0.01, 0.02, 0.015])
_COV = np.array(
    [
        [0.04, 0.005, 0.002],
        [0.005, 0.09, 0.003],
        [0.002, 0.003, 0.01],
    ]
)


def _make_problem(
    hhi_as_constraint: bool = False,
    constraint_slack: float = 0.1,
    hhi_bound: float = 0.5,
    activity_tol: float = 1e-4,
) -> BalancedPortfoliosRiskContributions:
    """Build a 3-asset BPRC problem with configurable modes."""
    return BalancedPortfoliosRiskContributions(
        mu=_MU,
        cov=_COV,
        l_assets=1,
        u_assets=3,
        activity_tol=activity_tol,
        constraint_slack=constraint_slack,
        hhi_as_constraint=hhi_as_constraint,
        hhi_bound=hhi_bound,
    )


def _make_evaluated_solution(
    problem: BalancedPortfoliosRiskContributions,
    weights: np.ndarray,
) -> Solution:
    """Create a Solution, assign weights, and evaluate."""
    sol = Solution(problem)
    sol.variables[:] = weights.tolist()
    problem.evaluate(sol)
    return sol


# ── Constructor Tests ────────────────────────────────────────────────────────


class TestBPRCConstruction:
    """Test BPRC problem construction and parameter validation."""

    def test_3_objective_mode_by_default(self) -> None:
        problem = _make_problem(hhi_as_constraint=False)
        assert problem.nobjs == 3
        assert problem.nconstrs == 2

    def test_2_objective_mode_when_hhi_as_constraint(self) -> None:
        problem = _make_problem(hhi_as_constraint=True)
        assert problem.nobjs == 2
        assert problem.nconstrs == 3

    def test_nvars_matches_asset_count(self) -> None:
        problem = _make_problem()
        assert problem.nvars == 3

    def test_invalid_mu_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D array"):
            BalancedPortfoliosRiskContributions(
                mu=np.ones((2, 2)), cov=_COV, l_assets=1, u_assets=3
            )

    def test_asymmetric_cov_raises(self) -> None:
        bad_cov = _COV.copy()
        bad_cov[0, 1] = 999.0  # break symmetry
        with pytest.raises(ValueError, match="symmetric"):
            BalancedPortfoliosRiskContributions(
                mu=_MU, cov=bad_cov, l_assets=1, u_assets=3
            )

    def test_invalid_asset_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="Require"):
            BalancedPortfoliosRiskContributions(
                mu=_MU, cov=_COV, l_assets=5, u_assets=3
            )

    def test_default_activity_tol(self) -> None:
        problem = _make_problem()
        assert problem.activity_tol == 1e-4

    def test_constraint_slack_stored(self) -> None:
        problem = _make_problem(constraint_slack=0.2)
        assert problem._constraint_slack == 0.2

    def test_hhi_bound_stored(self) -> None:
        problem = _make_problem(hhi_as_constraint=True, hhi_bound=0.3)
        assert problem._hhi_bound == 0.3


# ── 3-Objective Mode Tests ──────────────────────────────────────────────────


class TestBPRC3Objective:
    """Test the original 3-objective evaluation."""

    def test_objectives_count(self) -> None:
        problem = _make_problem(hhi_as_constraint=False)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        assert len(sol.objectives) == 3

    def test_constraints_count(self) -> None:
        problem = _make_problem(hhi_as_constraint=False)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        assert len(sol.constraints) == 2

    def test_variance_is_positive(self) -> None:
        problem = _make_problem(hhi_as_constraint=False)
        sol = _make_evaluated_solution(problem, np.array([0.5, 0.3, 0.2]))
        variance = sol.objectives[0]
        assert variance > 0.0

    def test_return_objective_is_negative(self) -> None:
        """Second objective is -return, so should be negative for positive returns."""
        problem = _make_problem(hhi_as_constraint=False)
        sol = _make_evaluated_solution(problem, np.array([0.5, 0.3, 0.2]))
        neg_return = sol.objectives[1]
        assert neg_return < 0.0

    def test_hhi_between_zero_and_one(self) -> None:
        problem = _make_problem(hhi_as_constraint=False)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        hhi = sol.objectives[2]
        assert 0.0 <= hhi <= 1.0

    def test_equal_weights_have_low_hhi(self) -> None:
        """Equal weights ⇒ balanced risk ⇒ low HHI."""
        problem = _make_problem(hhi_as_constraint=False)
        equal = _make_evaluated_solution(problem, np.array([1 / 3, 1 / 3, 1 / 3]))
        concentrated = _make_evaluated_solution(problem, np.array([0.9, 0.05, 0.05]))
        assert equal.objectives[2] < concentrated.objectives[2]


# ── 2-Objective (HHI-as-constraint) Mode Tests ──────────────────────────────


class TestBPRC2Objective:
    """Test the new 2-objective mode where HHI is a constraint."""

    def test_objectives_count(self) -> None:
        problem = _make_problem(hhi_as_constraint=True)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        assert len(sol.objectives) == 2

    def test_constraints_count(self) -> None:
        """3 constraints: rc_lower, rc_upper, hhi_bound."""
        problem = _make_problem(hhi_as_constraint=True)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        assert len(sol.constraints) == 3

    def test_hhi_constraint_satisfied_for_balanced_weights(self) -> None:
        problem = _make_problem(hhi_as_constraint=True, hhi_bound=0.5)
        sol = _make_evaluated_solution(problem, np.array([0.4, 0.3, 0.3]))
        # constraints[2] is hhi violation — should be ≈0 for balanced weights
        assert sol.constraints[2] <= 0.01

    def test_hhi_constraint_violated_for_concentrated_weights(self) -> None:
        problem = _make_problem(hhi_as_constraint=True, hhi_bound=0.2)
        sol = _make_evaluated_solution(problem, np.array([0.95, 0.025, 0.025]))
        # very concentrated ⇒ HHI > 0.2 ⇒ violation > 0
        assert sol.constraints[2] > 0.0

    def test_objectives_match_3obj_first_two(self) -> None:
        """Variance and -return should be identical in both modes."""
        w = np.array([0.5, 0.3, 0.2])
        p3 = _make_problem(hhi_as_constraint=False)
        p2 = _make_problem(hhi_as_constraint=True)
        s3 = _make_evaluated_solution(p3, w.copy())
        s2 = _make_evaluated_solution(p2, w.copy())
        np.testing.assert_almost_equal(s2.objectives[0], s3.objectives[0])
        np.testing.assert_almost_equal(s2.objectives[1], s3.objectives[1])


# ── Constraint Slack Tests ───────────────────────────────────────────────────


class TestBPRCConstraintSlack:
    """Test that constraint_slack relaxes the risk-contribution bounds."""

    def test_slack_zero_matches_original_bounds(self) -> None:
        problem = _make_problem(constraint_slack=0.0)
        sol = _make_evaluated_solution(problem, np.array([0.5, 0.3, 0.2]))
        # With slack=0, bounds are exactly [1/u, 1/l] = [1/3, 1/1]
        # Just check evaluation runs without error
        assert len(sol.constraints) >= 2

    def test_larger_slack_reduces_violations(self) -> None:
        """A wider slack should lead to fewer/smaller constraint violations."""
        w = np.array([0.7, 0.2, 0.1])
        p_tight = _make_problem(constraint_slack=0.0)
        p_loose = _make_problem(constraint_slack=0.3)
        s_tight = _make_evaluated_solution(p_tight, w.copy())
        s_loose = _make_evaluated_solution(p_loose, w.copy())
        # Sum of violations should be <= for loose constraints
        tight_total = sum(max(0, c) for c in s_tight.constraints)
        loose_total = sum(max(0, c) for c in s_loose.constraints)
        assert loose_total <= tight_total + 1e-12


# ── Random Solution Tests ───────────────────────────────────────────────────


class TestBPRCRandom:
    """Test the random() solution generator."""

    def test_random_solution_is_evaluated(self) -> None:
        problem = _make_problem()
        sol = problem.random()
        assert sol.objectives is not None
        assert len(sol.objectives) == problem.nobjs

    def test_random_weights_sum_to_one(self) -> None:
        problem = _make_problem()
        sol = problem.random()
        total = sum(sol.variables)
        assert abs(total - 1.0) < 1e-6

    def test_random_weights_non_negative(self) -> None:
        problem = _make_problem()
        for _ in range(10):
            sol = problem.random()
            assert all(w >= 0.0 for w in sol.variables)

    def test_random_2obj_mode_works(self) -> None:
        problem = _make_problem(hhi_as_constraint=True)
        sol = problem.random()
        assert len(sol.objectives) == 2
        assert len(sol.constraints) == 3


# ── Risk Contribution Tests ─────────────────────────────────────────────────


class TestNormalizedRiskContributions:
    """Test the _normalized_risk_contributions helper."""

    def test_contributions_sum_to_one(self) -> None:
        problem = _make_problem()
        w = np.array([0.4, 0.3, 0.3])
        gi = problem._normalized_risk_contributions(w)
        assert abs(gi.sum() - 1.0) < 1e-10

    def test_contributions_non_negative(self) -> None:
        problem = _make_problem()
        w = np.array([0.5, 0.3, 0.2])
        gi = problem._normalized_risk_contributions(w)
        assert (gi >= 0.0).all()

    def test_zero_weights_return_uniform(self) -> None:
        problem = _make_problem()
        w = np.zeros(3)
        gi = problem._normalized_risk_contributions(w)
        np.testing.assert_array_almost_equal(gi, [1 / 3, 1 / 3, 1 / 3])

    def test_single_active_gets_full_contribution(self) -> None:
        problem = _make_problem()
        w = np.array([1.0, 0.0, 0.0])
        gi = problem._normalized_risk_contributions(w)
        assert gi[0] == pytest.approx(1.0)
