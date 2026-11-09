"""Minimum variance at a configured return target on the feasible front."""

from __future__ import annotations

import math

import pandas as pd

from platypus import Solution

from .base_selection_strategy import BaseSelectionStrategy


class ReturnTargetSelection(BaseSelectionStrategy):
    """Interpolate from minimum-variance return to maximum feasible return.

    The target is based on the current BPRC candidates, not the unconstrained
    MV reference. Existing feasibility/risk-contribution bounds are retained.
    """

    def __init__(self, fraction: float = 0.75) -> None:
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Return target fraction must be between zero and one")
        self.fraction = fraction
        self.last_diagnostics: dict[str, float] = {}

    def select(self, front: list[Solution], asset_index: pd.Index) -> pd.Series:
        if not front or any(
            not math.isfinite(float(s.objectives[0]))
            or float(s.objectives[0]) < 0
            or not math.isfinite(float(s.objectives[1]))
            for s in front
        ):
            raise ValueError(
                "A nonempty front with finite return and variance is required"
            )
        minimum_risk = min(
            front, key=lambda s: (float(s.objectives[0]), float(s.objectives[1]))
        )
        lower = -float(minimum_risk.objectives[1])
        upper = max(-float(s.objectives[1]) for s in front)
        target = min(upper, lower + self.fraction * (upper - lower))
        eligible = [s for s in front if -float(s.objectives[1]) >= target]
        best = min(
            eligible, key=lambda s: (float(s.objectives[0]), float(s.objectives[1]))
        )
        self.last_diagnostics = {
            "return_target_fraction": self.fraction,
            "minimum_variance_expected_return": lower,
            "front_max_expected_return": upper,
            "target_expected_return": target,
            "selected_expected_return": -float(best.objectives[1]),
            "selected_volatility": math.sqrt(float(best.objectives[0])),
        }
        self.last_diagnostics.update(self._balance_diagnostics(best))
        result: pd.Series = self._normalise(best, asset_index)
        return result
