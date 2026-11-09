"""Unit tests for selection strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
from platypus import Solution

from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.selection_strategies import (
    MinVarianceSelection,
    RiskAversionSelection,
    SharpeSelection,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

SYMBOLS = pd.Index(["A", "B", "C"])
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
        alpha = i / max(n - 1, 1)
        w = np.zeros(3)
        w[0] = 1.0 - alpha
        w[1] = alpha
        w = w / w.sum()
        sol.variables[:] = w.tolist()
        problem.evaluate(sol)
        solutions.append(sol)
    return solutions


# ── SharpeSelection Tests ────────────────────────────────────────────────────


class TestSharpeSelection:
    def test_returns_series(self) -> None:
        front = _make_front()
        result = SharpeSelection().select(front, SYMBOLS)
        assert isinstance(result, pd.Series)
        assert list(result.index) == list(SYMBOLS)

    def test_weights_sum_to_one(self) -> None:
        front = _make_front()
        result = SharpeSelection().select(front, SYMBOLS)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_no_negative_weights(self) -> None:
        front = _make_front()
        result = SharpeSelection().select(front, SYMBOLS)
        assert (result >= -1e-8).all()


# ── MinVarianceSelection Tests ───────────────────────────────────────────────


class TestMinVarianceSelection:
    def test_returns_series(self) -> None:
        front = _make_front()
        result = MinVarianceSelection().select(front, SYMBOLS)
        assert isinstance(result, pd.Series)

    def test_selects_lowest_variance(self) -> None:
        front = _make_front()
        mv_result = MinVarianceSelection().select(front, SYMBOLS)
        sharpe_result = SharpeSelection().select(front, SYMBOLS)
        w_mv = mv_result.values
        w_sh = sharpe_result.values
        var_mv = float(w_mv @ _COV @ w_mv)
        var_sh = float(w_sh @ _COV @ w_sh)
        assert var_mv <= var_sh + 1e-8


# ── RiskAversionSelection Tests ──────────────────────────────────────────────


class TestRiskAversionSelection:
    def test_returns_series(self) -> None:
        front = _make_front()
        result = RiskAversionSelection(risk_aversion=0.5).select(front, SYMBOLS)
        assert isinstance(result, pd.Series)

    def test_high_risk_aversion_prefers_lower_variance(self) -> None:
        front = _make_front()
        low_ra = RiskAversionSelection(risk_aversion=0.1).select(front, SYMBOLS)
        high_ra = RiskAversionSelection(risk_aversion=10.0).select(front, SYMBOLS)
        var_low = float(low_ra.values @ _COV @ low_ra.values)
        var_high = float(high_ra.values @ _COV @ high_ra.values)
        assert var_high <= var_low + 1e-6

    def test_zero_risk_aversion_equals_max_return(self) -> None:
        front = _make_front()
        result = RiskAversionSelection(risk_aversion=0.0).select(front, SYMBOLS)
        ret = float(result.values @ _MU)
        max_ret = max(-float(s.objectives[1]) for s in front)
        assert abs(ret - max_ret) < 1e-4

    def test_weights_sum_to_one(self) -> None:
        front = _make_front()
        result = RiskAversionSelection(risk_aversion=0.5).select(front, SYMBOLS)
        assert abs(result.sum() - 1.0) < 1e-6
