"""Unit tests for comparison_plots module.

Verifies that plot functions produce files without errors.
Does not validate visual content — that is done by manual inspection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from platypus import NSGAII, Solution

from investment_tool.strategies.optimization_strategies.model.markowitz import (
    MarkowitzLongOnly,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_metrics import (
    compute_all_metrics,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.comparison_plots import (
    plot_metric_bars,
    plot_overlaid_fronts,
    plot_radar_chart,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_config import (
    SolverConfig,
)
from investment_tool.strategies.optimization_strategies.solver_comparison.solver_result import (
    SolverResult,
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


def _make_front(n: int = 10) -> list[Solution]:
    problem = MarkowitzLongOnly(mu=_MU, cov=_COV)
    solutions: list[Solution] = []
    for i in range(n):
        sol = Solution(problem)
        alpha = i / max(n - 1, 1)
        w = np.array([1.0 - alpha, alpha, 0.0])
        w = w / w.sum()
        sol.variables[:] = w.tolist()
        problem.evaluate(sol)
        solutions.append(sol)
    return solutions


def _make_result(label: str) -> SolverResult:
    front = _make_front(8)
    return SolverResult(
        config=SolverConfig(label=label, algorithm=NSGAII),
        front=front,
        all_archives=[front],
        wall_time_seconds=1.5,
        metrics=compute_all_metrics(front),
        best_weights=pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"]),
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPlotOverlaidFronts:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        path = plot_overlaid_fronts(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_with_reference(self, tmp_path: Path) -> None:
        results = [_make_result("A")]
        ref = _make_front(5)
        path = plot_overlaid_fronts(results, ref, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty_results(self) -> None:
        assert plot_overlaid_fronts([]) is None


class TestPlotMetricBars:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("X"), _make_result("Y")]
        path = plot_metric_bars(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty_results(self) -> None:
        assert plot_metric_bars([]) is None


class TestPlotRadarChart:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("R1"), _make_result("R2")]
        path = plot_radar_chart(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty_results(self) -> None:
        assert plot_radar_chart([]) is None
