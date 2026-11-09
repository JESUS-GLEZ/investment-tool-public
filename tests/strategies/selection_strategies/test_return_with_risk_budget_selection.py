"""Ensure higher-return selection respects a volatility, not variance, cap."""

import pandas as pd
import pytest
from platypus import Problem, Solution

from investment_tool.strategies.selection_strategies import (
    ReturnWithRiskBudgetSelection,
)


def candidate(variance: float, expected_return: float, asset: int) -> Solution:
    solution = Solution(Problem(3, 3))
    solution.variables[:] = [float(i == asset) for i in range(3)]
    solution.objectives[:] = [variance, -expected_return, 0.3]
    return solution


def test_selects_more_return_but_rejects_candidate_above_volatility_cap():
    front = [
        candidate(0.0001, 0.02, 0),
        candidate(0.000144, 0.023, 1),
        candidate(0.000196, 0.025, 2),
    ]
    selector = ReturnWithRiskBudgetSelection(0.25)
    weights = selector.select(front, pd.Index(["A", "B", "C"]))
    assert weights.to_dict() == {"A": 0, "B": 1, "C": 0}
    assert selector.reference_weights == {"A": 1.0}
    assert selector.last_diagnostics["selected_volatility"] == pytest.approx(0.012)
    assert selector.last_diagnostics["volatility_budget"] == pytest.approx(0.0125)
    assert selector.last_diagnostics["front_min_expected_return"] == pytest.approx(0.02)
    assert selector.last_diagnostics["front_max_expected_return"] == pytest.approx(
        0.025
    )
    assert (
        selector.last_diagnostics["selected_expected_return"]
        >= selector.last_diagnostics["sharpe_expected_return"]
    )


def test_accepts_exact_boundary_and_uses_squared_volatility_multiplier():
    front = [candidate(0.0001, 0.02, 0), candidate(0.0001 * 1.25**2, 0.024, 1)]
    weights = ReturnWithRiskBudgetSelection(0.25).select(
        front, pd.Index(["A", "B", "C"])
    )
    assert weights["B"] == 1


def test_zero_increase_keeps_the_reference_when_no_better_return_is_eligible():
    front = [candidate(0.0001, 0.02, 0), candidate(0.000144, 0.023, 1)]
    weights = ReturnWithRiskBudgetSelection(0).select(front, pd.Index(["A", "B", "C"]))
    assert weights["A"] == 1


def test_equal_returns_prefer_lower_variance():
    front = [candidate(0.0001, 0.02, 0), candidate(0.000144, 0.02, 1)]
    assert (
        ReturnWithRiskBudgetSelection().select(front, pd.Index(["A", "B", "C"]))["A"]
        == 1
    )


def test_zero_variance_reference_does_not_allow_positive_variance():
    front = [candidate(0, 0.01, 0), candidate(0.001, 0.02, 1)]
    assert (
        ReturnWithRiskBudgetSelection().select(front, pd.Index(["A", "B", "C"]))["A"]
        == 1
    )


@pytest.mark.parametrize("increase", [-0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_budget(increase):
    with pytest.raises(ValueError):
        ReturnWithRiskBudgetSelection(increase)


def test_rejects_invalid_front():
    selector = ReturnWithRiskBudgetSelection()
    with pytest.raises(ValueError, match="empty"):
        selector.select([], pd.Index(["A", "B", "C"]))
    with pytest.raises(ValueError, match="finite"):
        selector.select([candidate(float("nan"), 0.01, 0)], pd.Index(["A", "B", "C"]))
