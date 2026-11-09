import pandas as pd
import pytest
from platypus import Problem, Solution

from investment_tool.strategies.selection_strategies import ReturnTargetSelection


def candidate(variance, expected_return, asset):
    solution = Solution(Problem(4, 3))
    solution.variables[:] = [float(i == asset) for i in range(4)]
    solution.objectives[:] = [variance, -expected_return, 0.3]
    return solution


def test_target_uses_minimum_variance_anchor_and_selects_lowest_qualifying_risk():
    front = [
        candidate(0.01, 0.04, 0),
        candidate(0.03, 0.14, 1),
        candidate(0.04, 0.16, 2),
        candidate(0.05, 0.13, 3),
    ]
    selector = ReturnTargetSelection(0.75)
    weights = selector.select(front, pd.Index(list("ABCD")))
    assert weights["B"] == 1
    assert selector.last_diagnostics["target_expected_return"] == pytest.approx(0.13)
    assert (
        selector.last_diagnostics["selected_expected_return"]
        >= selector.last_diagnostics["target_expected_return"]
    )


@pytest.mark.parametrize("fraction,asset", [(0, "A"), (1, "C")])
def test_endpoints_match_minimum_variance_and_maximum_return(fraction, asset):
    front = [
        candidate(0.01, -0.05, 0),
        candidate(0.02, 0.01, 1),
        candidate(0.03, 0.08, 2),
    ]
    assert (
        ReturnTargetSelection(fraction).select(front, pd.Index(list("ABCD")))[asset]
        == 1
    )


def test_flat_return_range_uses_minimum_variance():
    front = [candidate(0.01, 0.02, 0), candidate(0.02, 0.02, 1)]
    assert ReturnTargetSelection().select(front, pd.Index(list("ABCD")))["A"] == 1


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError):
        ReturnTargetSelection(fraction)


def test_invalid_front_is_rejected():
    with pytest.raises(ValueError):
        ReturnTargetSelection().select([], pd.Index(list("ABCD")))
