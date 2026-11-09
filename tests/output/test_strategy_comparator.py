"""Unit tests for StrategyComparator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.output.strategy_comparator import StrategyComparator


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_strategy_results(
    n: int = 12,
    base_return: float = 0.01,
    noise_std: float = 0.005,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate synthetic backtest results for one strategy."""
    rng = np.random.RandomState(seed)
    results: list[dict[str, Any]] = []
    value = 10000.0
    for i in range(n):
        ret = base_return + rng.normal(0, noise_std) if i > 0 else 0.0
        value *= 1 + ret
        month = (i % 12) + 1
        year = 2024 + i // 12
        results.append(
            {
                "date": datetime(year, month, 15),
                "total_value": value,
                "return": ret,
            }
        )
    return results


def _make_two_strategies() -> dict[str, list[dict[str, Any]]]:
    """Create results for two strategies: a good one and a benchmark."""
    return {
        "buy_and_hold": _make_strategy_results(
            base_return=0.005, noise_std=0.01, seed=1
        ),
        "combined": _make_strategy_results(base_return=0.015, noise_std=0.02, seed=2),
    }


# ── Tests ───────────────────────────────────────────────────────────────────


class TestComputeComparisonMetrics:
    """Test metric computation."""

    def test_returns_dataframe(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        df = comparator._compute_comparison_metrics()
        assert isinstance(df, pd.DataFrame)

    def test_columns_are_strategy_names(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        df = comparator._compute_comparison_metrics()
        assert "buy_and_hold" in df.columns
        assert "combined" in df.columns

    def test_expected_metric_rows(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        df = comparator._compute_comparison_metrics()
        expected_metrics = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "calmar_ratio",
            "win_rate",
            "win_rate_vs_benchmark",
            "tracking_error",
            "information_ratio",
        }
        assert expected_metrics.issubset(set(df.index))

    def test_higher_return_strategy_has_higher_total_return(
        self, tmp_path: Path
    ) -> None:
        # Use many periods and low noise to make the outcome deterministic
        results = {
            "buy_and_hold": _make_strategy_results(
                n=36, base_return=0.005, noise_std=0.001, seed=1
            ),
            "combined": _make_strategy_results(
                n=36, base_return=0.020, noise_std=0.001, seed=2
            ),
        }
        comparator = StrategyComparator(
            results=results,
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
            benchmark_name="buy_and_hold",
        )
        df = comparator._compute_comparison_metrics()
        assert (
            df.loc["total_return", "combined"] > df.loc["total_return", "buy_and_hold"]
        )

    def test_benchmark_tracking_error_is_zero(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
            benchmark_name="buy_and_hold",
        )
        df = comparator._compute_comparison_metrics()
        assert abs(df.loc["tracking_error", "buy_and_hold"]) < 1e-10


class TestComparatorRun:
    """Test the full run() pipeline."""

    def test_run_creates_csv(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
            benchmark_name="buy_and_hold",
        )
        comparator.run()
        assert (tmp_path / "comparison_metrics.csv").exists()

    def test_run_creates_html_report(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
            benchmark_name="buy_and_hold",
        )
        comparator.run()
        assert (tmp_path / "comparison_report.html").exists()

    def test_run_returns_metrics_df(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        result = comparator.run()
        assert "metrics_df" in result
        assert isinstance(result["metrics_df"], pd.DataFrame)

    def test_run_returns_saved_files(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        result = comparator.run()
        assert "saved_files" in result
        assert len(result["saved_files"]) >= 2  # at least CSV + HTML

    def test_run_creates_plot_directory(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        comparator.run()
        assert (tmp_path / "comparison_plots").exists()

    def test_html_report_contains_strategy_names(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        comparator.run()
        content = (tmp_path / "comparison_report.html").read_text()
        assert "buy_and_hold" in content
        assert "combined" in content
        assert "Performance Metrics" in content

    def test_csv_is_readable(self, tmp_path: Path) -> None:
        comparator = StrategyComparator(
            results=_make_two_strategies(),
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        comparator.run()
        df = pd.read_csv(tmp_path / "comparison_metrics.csv", index_col=0)
        assert "sharpe_ratio" in df.index


class TestComparatorEdgeCases:
    """Test edge cases."""

    def test_single_strategy(self, tmp_path: Path) -> None:
        results = {"only_one": _make_strategy_results()}
        comparator = StrategyComparator(
            results=results,
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        result = comparator.run()
        assert "metrics_df" in result
        assert "only_one" in result["metrics_df"].columns

    def test_empty_results_no_crash(self, tmp_path: Path) -> None:
        results = {"empty": []}
        comparator = StrategyComparator(
            results=results,
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        result = comparator.run()
        assert "metrics_df" in result

    def test_benchmark_defaults_to_first(self, tmp_path: Path) -> None:
        results = _make_two_strategies()
        comparator = StrategyComparator(
            results=results,
            frequency=Frequency.MONTHLY,
            output_dir=tmp_path,
        )
        # First key is "buy_and_hold"
        assert comparator._benchmark == "buy_and_hold"
