"""Unit tests for the ScalarizedSubproblemSolver.

Tests both scalarization methods (weighted sum and augmented Tchebycheff),
constraint handling, warm-start initialisation, and edge cases.
"""

from __future__ import annotations

import numpy as np

from investment_tool.strategies.optimization_strategies.scalarized_solver import (
    ScalarizationMethod,
    ScalarizedSubproblemSolver,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

_MU = np.array([0.01, 0.02, 0.015])
_COV = np.array(
    [
        [0.04, 0.005, 0.002],
        [0.005, 0.09, 0.003],
        [0.002, 0.003, 0.01],
    ]
)


def _make_solver(
    method: ScalarizationMethod = ScalarizationMethod.AUGMENTED_TCHEBYCHEFF,
    hhi_as_constraint: bool = False,
    ideal: np.ndarray | None = None,
    nadir: np.ndarray | None = None,
) -> ScalarizedSubproblemSolver:
    """Build a 3-asset solver for tests."""
    return ScalarizedSubproblemSolver(
        mu=_MU,
        cov=_COV,
        l_assets=1,
        u_assets=3,
        method=method,
        hhi_as_constraint=hhi_as_constraint,
        ideal=ideal,
        nadir=nadir,
    )


# ── Construction Tests ───────────────────────────────────────────────────────


class TestSolverConstruction:
    """Test ScalarizedSubproblemSolver construction."""

    def test_3_objective_mode_by_default(self) -> None:
        solver = _make_solver(hhi_as_constraint=False)
        assert solver.n_objectives == 3

    def test_2_objective_mode_when_hhi_constraint(self) -> None:
        solver = _make_solver(hhi_as_constraint=True)
        assert solver.n_objectives == 2

    def test_stores_mu_and_cov(self) -> None:
        solver = _make_solver()
        np.testing.assert_array_equal(solver.mu, _MU)
        np.testing.assert_array_equal(solver.cov, _COV)

    def test_stores_asset_bounds(self) -> None:
        solver = _make_solver()
        assert solver.l_assets == 1
        assert solver.u_assets == 3


# ── Raw Objectives Tests ────────────────────────────────────────────────────


class TestRawObjectives:
    """Test raw_objectives computation."""

    def test_3_objective_returns_three_values(self) -> None:
        solver = _make_solver(hhi_as_constraint=False)
        w = np.array([0.4, 0.3, 0.3])
        objs = solver.raw_objectives(w)
        assert objs.shape == (3,)

    def test_2_objective_returns_two_values(self) -> None:
        solver = _make_solver(hhi_as_constraint=True)
        w = np.array([0.4, 0.3, 0.3])
        objs = solver.raw_objectives(w)
        assert objs.shape == (2,)

    def test_variance_is_positive(self) -> None:
        solver = _make_solver()
        w = np.array([0.5, 0.3, 0.2])
        objs = solver.raw_objectives(w)
        assert objs[0] > 0.0

    def test_negative_return_is_negative(self) -> None:
        solver = _make_solver()
        w = np.array([0.5, 0.3, 0.2])
        objs = solver.raw_objectives(w)
        assert objs[1] < 0.0  # -return is negative for positive mu

    def test_hhi_between_zero_and_one(self) -> None:
        solver = _make_solver(hhi_as_constraint=False)
        w = np.array([0.4, 0.3, 0.3])
        objs = solver.raw_objectives(w)
        assert 0.0 <= objs[2] <= 1.0


# ── Scalarization Tests ─────────────────────────────────────────────────────


class TestScalarization:
    """Test scalarization methods."""

    def test_weighted_sum_returns_scalar(self) -> None:
        solver = _make_solver(method=ScalarizationMethod.WEIGHTED_SUM)
        w = np.array([0.4, 0.3, 0.3])
        lam = np.array([1.0, 1.0, 1.0])
        val = solver._scalarize(w, lam)
        assert isinstance(val, float)

    def test_tchebycheff_returns_scalar(self) -> None:
        solver = _make_solver(method=ScalarizationMethod.AUGMENTED_TCHEBYCHEFF)
        w = np.array([0.4, 0.3, 0.3])
        lam = np.array([1.0, 1.0, 1.0])
        val = solver._scalarize(w, lam)
        assert isinstance(val, float)

    def test_different_lambdas_produce_different_values(self) -> None:
        solver = _make_solver(
            method=ScalarizationMethod.WEIGHTED_SUM,
            ideal=np.array([0.0, -0.03, 0.0]),
            nadir=np.array([0.1, 0.0, 1.0]),
        )
        w = np.array([0.4, 0.3, 0.3])
        lam1 = np.array([1.0, 0.0, 0.0])
        lam2 = np.array([0.0, 1.0, 0.0])
        assert solver._scalarize(w, lam1) != solver._scalarize(w, lam2)


# ── Solve Tests ──────────────────────────────────────────────────────────────


class TestSolve:
    """Test the solve() method — core SLSQP integration."""

    def test_returns_valid_weights(self) -> None:
        solver = _make_solver(
            ideal=np.array([0.0, -0.03, 0.0]),
            nadir=np.array([0.1, 0.0, 1.0]),
        )
        lam = np.array([1.0, 1.0, 1.0])
        w, val, converged = solver.solve(lam, n_random_starts=2, maxiter=200)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= 0.0).all()

    def test_weights_sum_to_one(self) -> None:
        solver = _make_solver(
            ideal=np.array([0.0, -0.03, 0.0]),
            nadir=np.array([0.1, 0.0, 1.0]),
        )
        lam = np.array([0.5, 0.5, 0.0])
        w, _, _ = solver.solve(lam, n_random_starts=2, maxiter=200)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_warm_start_improves_or_equals(self) -> None:
        """A warm-start from a good solution should not degrade quality."""
        solver = _make_solver(
            ideal=np.array([0.0, -0.03, 0.0]),
            nadir=np.array([0.1, 0.0, 1.0]),
        )
        lam = np.array([1.0, 1.0, 1.0])
        w_cold, val_cold, _ = solver.solve(lam, n_random_starts=3, maxiter=200)
        w_warm, val_warm, _ = solver.solve(
            lam, x0=w_cold, n_random_starts=1, maxiter=200
        )
        # Warm start should be at least as good (within tolerance)
        assert val_warm <= val_cold + 1e-6

    def test_2_objective_mode_solve(self) -> None:
        solver = _make_solver(
            hhi_as_constraint=True,
            ideal=np.array([0.0, -0.03]),
            nadir=np.array([0.1, 0.0]),
        )
        lam = np.array([0.5, 0.5])
        w, val, converged = solver.solve(lam, n_random_starts=2, maxiter=200)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= 0.0).all()

    def test_variance_focused_produces_low_variance(self) -> None:
        """Lambda focused on variance should yield a low-variance portfolio."""
        solver = _make_solver(
            hhi_as_constraint=False,
            ideal=np.array([0.0, -0.03, 0.0]),
            nadir=np.array([0.1, 0.0, 1.0]),
        )
        lam_var = np.array([1.0, 0.0, 0.0])
        lam_ret = np.array([0.0, 1.0, 0.0])
        w_var, _, _ = solver.solve(lam_var, n_random_starts=3, maxiter=300)
        w_ret, _, _ = solver.solve(lam_ret, n_random_starts=3, maxiter=300)
        var_var = float(w_var @ _COV @ w_var)
        var_ret = float(w_ret @ _COV @ w_ret)
        assert var_var <= var_ret + 1e-4


# ── Risk Contribution Constraint Tests ───────────────────────────────────────


class TestRiskContributionConstraints:
    """Test that SLSQP respects risk-contribution constraints."""

    def test_risk_contributions_sum_to_one(self) -> None:
        solver = _make_solver()
        w = np.array([0.4, 0.3, 0.3])
        gi = solver._normalized_risk_contributions(w)
        assert abs(gi.sum() - 1.0) < 1e-10

    def test_risk_contributions_non_negative(self) -> None:
        solver = _make_solver()
        w = np.array([0.5, 0.3, 0.2])
        gi = solver._normalized_risk_contributions(w)
        assert (gi >= 0.0).all()

    def test_zero_weights_return_uniform(self) -> None:
        solver = _make_solver()
        w = np.zeros(3)
        gi = solver._normalized_risk_contributions(w)
        np.testing.assert_array_almost_equal(gi, [1 / 3, 1 / 3, 1 / 3])


# ── ScalarizationMethod Enum Tests ──────────────────────────────────────────


class TestScalarizationMethodEnum:
    """Test the ScalarizationMethod enum."""

    def test_weighted_sum_value(self) -> None:
        assert ScalarizationMethod.WEIGHTED_SUM.value == "weighted_sum"

    def test_tchebycheff_value(self) -> None:
        assert (
            ScalarizationMethod.AUGMENTED_TCHEBYCHEFF.value == "augmented_tchebycheff"
        )

    def test_from_string(self) -> None:
        assert ScalarizationMethod("weighted_sum") == ScalarizationMethod.WEIGHTED_SUM
