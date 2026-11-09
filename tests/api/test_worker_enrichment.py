"""Tests for worker enrichment functions added in the API layer.

Covers _add_prediction_figures, _add_comparison_figures, _add_solver_figures,
and _add_comparison_metrics — all using lightweight in-process mocks so no
coordinators, Redis, or Celery are needed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predictor_result(label: str, n_dates: int = 30, n_assets: int = 3) -> Any:
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    pred = pd.DataFrame(
        np.random.randn(n_dates, n_assets) * 0.01, index=dates, columns=assets
    )
    act = pd.DataFrame(
        np.random.randn(n_dates, n_assets) * 0.01, index=dates, columns=assets
    )
    per_asset = pd.DataFrame(
        {
            "directional_accuracy": np.random.rand(n_assets),
            "mae": np.random.rand(n_assets) * 0.01,
        },
        index=assets,
    )
    return SimpleNamespace(
        config=SimpleNamespace(label=label),
        predictions_df=pred,
        actuals_df=act,
        per_asset_metrics=per_asset,
        metrics={
            "mae": 0.01,
            "rmse": 0.015,
            "directional_accuracy": 0.55,
            "information_coefficient": 0.1,
            "hit_rate_top_k": 0.6,
        },
        wall_time_seconds=1.5,
    )


def _make_solver_result(label: str, n_solutions: int = 10) -> Any:
    front = [
        SimpleNamespace(objectives=[np.random.rand() * 0.01, -np.random.rand() * 0.1])
        for _ in range(n_solutions)
    ]
    return SimpleNamespace(
        config=SimpleNamespace(label=label),
        front=front,
        wall_time_seconds=2.0,
        metrics={
            "hypervolume": np.random.rand(),
            "best_sharpe": np.random.rand() * 2,
            "median_sharpe": np.random.rand() * 1.5,
            "spread": np.random.rand(),
            "cardinality": float(np.random.randint(5, 15)),
        },
    )


def _make_snapshot_list(n: int = 50) -> list[dict[str, Any]]:
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    returns = np.random.default_rng(42).normal(0, 0.01, n)
    vals = np.cumprod(1 + returns) * 10000
    return [
        {"date": str(d.date()), "total_value": float(v), "return": float(r)}
        for d, v, r in zip(dates, vals, returns)
    ]


# ---------------------------------------------------------------------------
# _add_prediction_figures
# ---------------------------------------------------------------------------


class TestAddPredictionFigures:
    def test_produces_five_figures(self) -> None:
        from api.result_presenter import _add_prediction_figures

        raw: dict[str, Any] = {
            "results": [
                _make_predictor_result("ModelA"),
                _make_predictor_result("ModelB"),
            ]
        }
        out = _add_prediction_figures(raw)
        assert len(out["figures"]) == 5

    def test_undefined_ranking_metrics_do_not_become_zero_bars(self) -> None:
        from api.result_presenter import _add_prediction_figures

        result = _make_predictor_result("ZeroReturn")
        result.metrics["information_coefficient"] = float("nan")
        result.metrics["hit_rate_top_k"] = float("nan")
        figures = _add_prediction_figures({"results": [result]})["figures"]
        bars = figures[2]["data"]
        assert bars[3]["y"] == [None]
        assert bars[4]["y"] == [None]
        assert bars[3]["text"] == ["N/A"]

    def test_figure_titles_present(self) -> None:
        from api.result_presenter import _add_prediction_figures

        raw: dict[str, Any] = {"results": [_make_predictor_result("X")]}
        out = _add_prediction_figures(raw)
        titles = [f["layout"]["title"]["text"] for f in out["figures"]]
        assert "Predicted vs. Actual Returns" in titles
        assert "Prediction Error Distributions" in titles
        assert "Directional Accuracy Over Time" in titles

    def test_returns_unchanged_when_no_results(self) -> None:
        from api.result_presenter import _add_prediction_figures

        raw: dict[str, Any] = {}
        out = _add_prediction_figures(raw)
        assert "figures" not in out

    def test_figures_are_dicts(self) -> None:
        from api.result_presenter import _add_prediction_figures

        raw: dict[str, Any] = {"results": [_make_predictor_result("M")]}
        out = _add_prediction_figures(raw)
        for fig in out["figures"]:
            assert isinstance(fig, dict)
            assert "data" in fig
            assert "layout" in fig

    def test_handles_empty_per_asset_metrics(self) -> None:
        from api.result_presenter import _add_prediction_figures

        r = _make_predictor_result("NoAssets")
        r.per_asset_metrics = pd.DataFrame()
        raw: dict[str, Any] = {"results": [r]}
        # Should not raise; heatmap will be skipped, 4 figures remain
        out = _add_prediction_figures(raw)
        assert len(out.get("figures", [])) >= 3


# ---------------------------------------------------------------------------
# _add_solver_figures
# ---------------------------------------------------------------------------


class TestAddSolverFigures:
    def test_produces_three_figures(self) -> None:
        from api.result_presenter import _add_solver_figures

        raw: dict[str, Any] = {
            "results": [_make_solver_result("NSGAII"), _make_solver_result("SPEA2")]
        }
        out = _add_solver_figures(raw)
        assert len(out["figures"]) == 3

    def test_figure_titles(self) -> None:
        from api.result_presenter import _add_solver_figures

        raw: dict[str, Any] = {
            "results": [_make_solver_result("A"), _make_solver_result("B")]
        }
        out = _add_solver_figures(raw)
        titles = [f["layout"]["title"]["text"] for f in out["figures"]]
        assert "Solver Comparison — Pareto Fronts" in titles
        assert "Solver Comparison — Quality Metrics" in titles
        assert "Solver Comparison — Radar" in titles

    def test_returns_unchanged_when_no_results(self) -> None:
        from api.result_presenter import _add_solver_figures

        raw: dict[str, Any] = {}
        out = _add_solver_figures(raw)
        assert "figures" not in out

    def test_single_solver(self) -> None:
        from api.result_presenter import _add_solver_figures

        raw: dict[str, Any] = {"results": [_make_solver_result("OnlySolver")]}
        out = _add_solver_figures(raw)
        assert len(out["figures"]) == 3


# ---------------------------------------------------------------------------
# _add_comparison_figures (strategy comparison)
# ---------------------------------------------------------------------------


class TestAddComparisonFigures:
    def test_uses_monthly_frequency_and_configured_risk_free_rate(self) -> None:
        from api.result_presenter import _add_comparison_figures
        from investment_tool.constants import Frequency
        from investment_tool.output.metrics import MetricsCalculator

        rows = _make_snapshot_list(12)
        raw = {
            "frequency": "monthly",
            "risk_free_rate": 0.05,
            "individual_results": {
                "strategy": {"results": rows, "portfolio_history": []}
            },
        }
        enriched = _add_comparison_figures(raw)
        actual = enriched["individual_results"]["strategy"]["metrics"]
        expected = MetricsCalculator(rows, [], Frequency.MONTHLY, 0.05).compute_all()
        assert actual["sharpe_ratio"] == pytest.approx(expected["sharpe_ratio"])
        assert actual["annualized_return"] == pytest.approx(
            expected["annualized_return"]
        )
        assert actual["metric_definitions"]["periods_per_year"] == 12

    def test_unknown_frequency_does_not_invent_annualized_metrics(self) -> None:
        from api.result_presenter import _add_comparison_figures

        raw = {"individual_results": {"strategy": {"results": _make_snapshot_list()}}}
        enriched = _add_comparison_figures(raw)
        assert "figures" in enriched
        assert "metrics" not in enriched["individual_results"]["strategy"]

    def test_produces_figures_from_raw_snapshots(self) -> None:
        from api.result_presenter import _add_comparison_figures

        raw: dict[str, Any] = {
            "individual_results": {
                "StratA": {"results": _make_snapshot_list(), "portfolio_history": []},
                "StratB": {"results": _make_snapshot_list(), "portfolio_history": []},
            }
        }
        out = _add_comparison_figures(raw)
        assert "figures" in out
        assert len(out["figures"]) > 0

    def test_figure_titles_prefixed_with_strategy_name(self) -> None:
        from api.result_presenter import _add_comparison_figures

        raw: dict[str, Any] = {
            "individual_results": {
                "MyStrategy": {
                    "results": _make_snapshot_list(),
                    "portfolio_history": [],
                },
            }
        }
        out = _add_comparison_figures(raw)
        titles = [f["layout"]["title"]["text"] for f in out["figures"]]
        assert all("MyStrategy —" in t for t in titles)

    def test_returns_unchanged_when_no_individual_results(self) -> None:
        from api.result_presenter import _add_comparison_figures

        raw: dict[str, Any] = {}
        out = _add_comparison_figures(raw)
        assert "figures" not in out

    def test_skips_entry_with_empty_snapshots(self) -> None:
        from api.result_presenter import _add_comparison_figures

        raw: dict[str, Any] = {
            "individual_results": {
                "Good": {"results": _make_snapshot_list(), "portfolio_history": []},
                "Empty": {"results": [], "portfolio_history": []},
            }
        }
        out = _add_comparison_figures(raw)
        titles = [f["layout"]["title"]["text"] for f in out.get("figures", [])]
        assert all("Good —" in t for t in titles)
        assert not any("Empty —" in t for t in titles)


# ---------------------------------------------------------------------------
# _add_comparison_metrics
# ---------------------------------------------------------------------------


class TestAddComparisonMetrics:
    def test_transposes_metrics_df(self) -> None:
        from api.result_presenter import _add_comparison_metrics

        df = pd.DataFrame(
            {"StratA": [0.1, 0.05], "StratB": [0.08, 0.03]},
            index=["annualized_return", "max_drawdown"],
        )
        raw: dict[str, Any] = {"metrics_df": df}
        out = _add_comparison_metrics(raw)
        assert "comparison_metrics" in out
        assert "StratA" in out["comparison_metrics"]
        assert out["comparison_metrics"]["StratA"][
            "annualized_return"
        ] == pytest.approx(0.1)

    def test_uses_summary_table_as_fallback(self) -> None:
        from api.result_presenter import _add_comparison_metrics

        df = pd.DataFrame(
            {"SolverX": [1.5]},
            index=["best_sharpe"],
        )
        raw: dict[str, Any] = {"summary_table": df}
        out = _add_comparison_metrics(raw)
        assert "SolverX" in out["comparison_metrics"]

    def test_nan_values_become_none(self) -> None:
        from api.result_presenter import _add_comparison_metrics

        df = pd.DataFrame(
            {"StratA": [float("nan")]},
            index=["broken_metric"],
        )
        raw: dict[str, Any] = {"metrics_df": df}
        out = _add_comparison_metrics(raw)
        assert out["comparison_metrics"]["StratA"]["broken_metric"] is None

    def test_returns_unchanged_when_no_df(self) -> None:
        from api.result_presenter import _add_comparison_metrics

        raw: dict[str, Any] = {"something_else": 42}
        out = _add_comparison_metrics(raw)
        assert "comparison_metrics" not in out


def test_solver_plot_includes_reference_and_seed_baseline():
    from api.result_presenter import _add_solver_figures

    result = _make_solver_result("NSGAII")
    raw = {
        "results": [result],
        "reference_front": result.front,
        "seed_front": result.front,
    }
    data = _add_solver_figures(raw)["figures"][0]["data"]
    assert len(data) == 3
    assert data[1]["name"] == "MV reference (long-only, no BPRC constraints)"
    assert data[1]["mode"] == "lines"
    assert data[2]["name"] == "Multi-Greedy RP seeds"
