"""Unit tests for comparison_metrics module."""

from __future__ import annotations

import numpy as np
from platypus import Solution

from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_metrics import (
    best_sharpe,
    compute_all_metrics,
    front_cardinality,
    hypervolume,
    median_sharpe,
    spacing,
    spread,
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


def _make_front(n: int = 10) -> list[Solution]:
    """Generate a small evaluated Markowitz front for testing."""
    problem = MarkowitzLongOnly(mu=_MU, cov=_COV)
    solutions: list[Solution] = []
    for i in range(n):
        sol = Solution(problem)
        w = np.zeros(3)
        # Linearly interpolate between asset 0 and asset 1
        alpha = i / max(n - 1, 1)
        w[0] = 1.0 - alpha
        w[1] = alpha
        w = w / w.sum()
        sol.variables[:] = w.tolist()
        problem.evaluate(sol)
        solutions.append(sol)
    return solutions


# ── Hypervolume Tests ────────────────────────────────────────────────────────


class TestHypervolume:
    """Test hypervolume computation."""

    def test_empty_front(self) -> None:
        assert hypervolume([], np.array([1.0, 1.0])) == 0.0

    def test_positive_for_valid_front(self) -> None:
        front = _make_front(10)
        ref = np.array([0.5, 0.5])
        hv = hypervolume(front, ref)
        assert hv > 0.0

    def test_monotonic_with_more_solutions(self) -> None:
        """More solutions → hypervolume should not decrease."""
        front_small = _make_front(3)
        front_large = _make_front(10)
        ref = np.array([0.5, 0.5])
        assert hypervolume(front_large, ref) >= hypervolume(front_small, ref)

    def test_zero_when_ref_dominated(self) -> None:
        """Ref point inside the front → 0 hypervolume."""
        front = _make_front(5)
        # Put ref at the ideal point (dominated by nothing)
        ref = np.array([0.0, -1.0])
        assert hypervolume(front, ref) == 0.0


# ── Spacing Tests ────────────────────────────────────────────────────────────


class TestSpacing:
    """Test spacing (uniformity) metric."""

    def test_empty_front(self) -> None:
        assert spacing([]) == 0.0

    def test_single_solution(self) -> None:
        assert spacing(_make_front(1)) == 0.0

    def test_non_negative(self) -> None:
        assert spacing(_make_front(10)) >= 0.0


# ── Spread Tests ─────────────────────────────────────────────────────────────


class TestSpread:
    """Test spread (extent) metric."""

    def test_empty_front(self) -> None:
        assert spread([]) == 0.0

    def test_positive_for_diverse_front(self) -> None:
        assert spread(_make_front(10)) > 0.0


# ── Cardinality Tests ────────────────────────────────────────────────────────


class TestCardinality:
    """Test front cardinality."""

    def test_counts_correctly(self) -> None:
        assert front_cardinality(_make_front(7)) == 7

    def test_empty(self) -> None:
        assert front_cardinality([]) == 0


# ── Sharpe Tests ─────────────────────────────────────────────────────────────


class TestSharpeMetrics:
    """Test best_sharpe and median_sharpe."""

    def test_best_sharpe_positive(self) -> None:
        front = _make_front(10)
        assert best_sharpe(front) > 0.0

    def test_median_sharpe_positive(self) -> None:
        front = _make_front(10)
        assert median_sharpe(front) > 0.0

    def test_best_ge_median(self) -> None:
        front = _make_front(10)
        assert best_sharpe(front) >= median_sharpe(front)

    def test_empty_best_sharpe(self) -> None:
        assert best_sharpe([]) == float("-inf")

    def test_empty_median_sharpe(self) -> None:
        assert median_sharpe([]) == 0.0


# ── compute_all_metrics Tests ────────────────────────────────────────────────


class TestComputeAllMetrics:
    """Test the aggregate metrics function."""

    def test_returns_all_keys(self) -> None:
        front = _make_front(10)
        metrics = compute_all_metrics(front)
        expected_keys = {
            "hypervolume",
            "spacing",
            "spread",
            "cardinality",
            "best_sharpe",
            "median_sharpe",
        }
        assert set(metrics.keys()) == expected_keys

    def test_all_values_finite(self) -> None:
        front = _make_front(10)
        metrics = compute_all_metrics(front)
        for key, val in metrics.items():
            assert np.isfinite(val), f"{key} is not finite: {val}"

    def test_custom_ref_point(self) -> None:
        front = _make_front(10)
        ref = np.array([1.0, 1.0])
        metrics = compute_all_metrics(front, ref_point=ref)
        assert metrics["hypervolume"] > 0.0

    def test_empty_front(self) -> None:
        metrics = compute_all_metrics([])
        assert metrics["cardinality"] == 0.0
        assert metrics["hypervolume"] == 0.0
