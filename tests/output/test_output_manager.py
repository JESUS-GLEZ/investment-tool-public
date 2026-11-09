"""Tests for OutputManager."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.output.output_manager import OutputManager


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_results(n: int = 12) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    value = 10000.0
    for i in range(n):
        ret = 0.01 if i > 0 else 0.0
        value *= 1 + ret
        month = (i % 12) + 1
        year = 2024 + i // 12
        results.append(
            {
                "date": datetime(year, month, 15),
                "total_value": value,
                "return": ret,
                "selected_assets": {"AAPL": 0.6, "GOOG": 0.4},
                "asset_prices": {"AAPL": 150.0 + i, "GOOG": 200.0 + i},
            }
        )
    return results


def _make_portfolio_history(n: int = 12) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for i in range(n):
        month = (i % 12) + 1
        year = 2024 + i // 12
        history.append(
            {
                "date": datetime(year, month, 15),
                "cash": 500.0,
                "positions": {"AAPL": 10.0, "GOOG": 5.0},
                "trades": {"AAPL": 1.0},
                "prices": {"AAPL": 150.0, "GOOG": 200.0},
                "pre_rebalance_value": 10000.0 + i * 100,
                "post_rebalance_value": 10050.0 + i * 100,
            }
        )
    return history


# ── Tests ───────────────────────────────────────────────────────────────────


class TestBacktestOutput:
    """Test OutputManager.generate_backtest_output."""

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_png_plots=False,
        )
        output = manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        assert "metrics" in output
        assert "results_df" in output
        assert "saved_files" in output
        assert "output_dir" in output

    def test_metrics_are_comprehensive(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        output = manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        metrics = output["metrics"]
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "win_rate" in metrics
        assert "total_return" in metrics

    def test_results_df_is_dataframe(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        output = manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        assert isinstance(output["results_df"], pd.DataFrame)

    def test_creates_csv_files(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        assert (tmp_path / "backtest_results.csv").exists()
        assert (tmp_path / "backtest_metrics.csv").exists()

    def test_creates_html_report(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=True,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        assert (tmp_path / "report.html").exists()

    def test_creates_plots_html(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=True,
            save_png_plots=False,
            generate_report=False,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        plots_dir = tmp_path / "plots"
        assert plots_dir.exists()
        html_files = list(plots_dir.glob("*.html"))
        assert len(html_files) > 0

    def test_no_plots_when_disabled(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        plots_dir = tmp_path / "plots"
        if plots_dir.exists():
            html_files = list(plots_dir.glob("*.html"))
            assert len(html_files) == 0


class TestLiveOutput:
    """Test OutputManager.generate_live_output."""

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
        )
        weights = pd.Series([0.5, 0.3, 0.2], index=["AAPL", "GOOG", "MSFT"])
        output = manager.generate_live_output(optimal_weights=weights)
        assert "concentration" in output
        assert "optimal_weights" in output
        assert "saved_files" in output

    def test_concentration_metrics(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
        )
        weights = pd.Series([0.5, 0.3, 0.2], index=["AAPL", "GOOG", "MSFT"])
        output = manager.generate_live_output(optimal_weights=weights)
        concentration = output["concentration"]
        assert "hhi" in concentration
        assert "num_active_positions" in concentration
        assert concentration["num_active_positions"] == 3

    def test_creates_weight_files(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
        )
        weights = pd.Series([0.6, 0.4], index=["AAPL", "GOOG"])
        manager.generate_live_output(optimal_weights=weights)
        assert (tmp_path / "optimal_weights.csv").exists()
        assert (tmp_path / "optimal_weights.json").exists()

    def test_creates_portfolio_snapshot(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
        )
        weights = pd.Series([0.5, 0.5], index=["AAPL", "GOOG"])
        snapshot = {"cash": 1000.0, "positions": {"AAPL": 10}}
        manager.generate_live_output(
            optimal_weights=weights,
            portfolio_snapshot=snapshot,
        )
        assert (tmp_path / "portfolio_snapshot.json").exists()


# ── Rebalancing Detail Tests ────────────────────────────────────────────────


class TestBuildRebalancingDetail:
    """Test OutputManager._build_rebalancing_detail."""

    def test_produces_expected_columns(self) -> None:
        results = _make_results(3)
        df = OutputManager._build_rebalancing_detail(results)
        assert set(df.columns) >= {
            "date",
            "symbol",
            "weight",
            "price",
            "next_price",
            "price_change_pct",
        }

    def test_row_count_matches_selected_assets(self) -> None:
        results = _make_results(3)
        df = OutputManager._build_rebalancing_detail(results)
        # 3 periods × 2 assets = 6 rows
        assert len(df) == 6

    def test_next_price_populated_for_non_last_periods(self) -> None:
        results = _make_results(3)
        df = OutputManager._build_rebalancing_detail(results)
        # First two periods should have next_price
        first_period = df[df["date"] == pd.Timestamp(results[0]["date"])]
        assert first_period["next_price"].notna().all()

    def test_last_period_has_no_next_price(self) -> None:
        results = _make_results(3)
        df = OutputManager._build_rebalancing_detail(results)
        last_period = df[df["date"] == pd.Timestamp(results[-1]["date"])]
        assert last_period["next_price"].isna().all()

    def test_price_change_pct_correct(self) -> None:
        results = [
            {
                "date": datetime(2024, 1, 15),
                "total_value": 10000.0,
                "return": 0.0,
                "selected_assets": {"AAPL": 1.0},
                "asset_prices": {"AAPL": 100.0},
            },
            {
                "date": datetime(2024, 2, 15),
                "total_value": 10500.0,
                "return": 0.05,
                "selected_assets": {"AAPL": 1.0},
                "asset_prices": {"AAPL": 110.0},
            },
        ]
        df = OutputManager._build_rebalancing_detail(results)
        first_row = df.iloc[0]
        assert abs(first_row["price_change_pct"] - 10.0) < 0.01

    def test_empty_results_returns_empty_df(self) -> None:
        df = OutputManager._build_rebalancing_detail([])
        assert df.empty
        assert "symbol" in df.columns

    def test_results_without_selected_assets_skipped(self) -> None:
        results = [
            {"date": datetime(2024, 1, 15), "total_value": 10000.0, "return": 0.0},
        ]
        df = OutputManager._build_rebalancing_detail(results)
        assert df.empty


class TestBuildResultsDfEnriched:
    """Test that _build_results_df includes the selected_assets_summary."""

    def test_selected_assets_summary_column_present(self) -> None:
        results = _make_results(2)
        df = OutputManager._build_results_df(results)
        assert "selected_assets_summary" in df.columns

    def test_selected_assets_summary_content(self) -> None:
        results = _make_results(1)
        df = OutputManager._build_results_df(results)
        summary = df["selected_assets_summary"].iloc[0]
        assert "AAPL" in summary
        assert "GOOG" in summary
        assert "60.00%" in summary

    def test_nested_dicts_dropped(self) -> None:
        results = _make_results(1)
        df = OutputManager._build_results_df(results)
        assert "selected_assets" not in df.columns
        assert "asset_prices" not in df.columns


class TestRebalancingDetailExport:
    """Test that rebalancing_detail.csv is created during backtest output."""

    def test_rebalancing_detail_csv_created(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        assert (tmp_path / "rebalancing_detail.csv").exists()

    def test_rebalancing_detail_csv_has_correct_columns(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=False,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        df = pd.read_csv(tmp_path / "rebalancing_detail.csv")
        for col in [
            "date",
            "symbol",
            "weight",
            "price",
            "next_price",
            "price_change_pct",
        ]:
            assert col in df.columns


class TestRebalancingDetailInReport:
    """Test that the HTML report includes the rebalancing detail section."""

    def test_report_contains_rebalancing_section(self, tmp_path: Path) -> None:
        manager = OutputManager(
            output_dir=tmp_path,
            frequency=Frequency.MONTHLY,
            save_html_plots=False,
            save_png_plots=False,
            generate_report=True,
        )
        manager.generate_backtest_output(
            results=_make_results(),
            portfolio_history=_make_portfolio_history(),
        )
        content = (tmp_path / "report.html").read_text()
        assert "Rebalancing Detail" in content
        assert "AAPL" in content
        assert "GOOG" in content
