"""Tests for DataExporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from investment_tool.output.export import DataExporter


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def results_df() -> pd.DataFrame:
    dates = pd.date_range("2024-01-15", periods=6, freq="ME")
    return pd.DataFrame(
        {
            "total_value": [10000, 10100, 10200, 10300, 10400, 10500],
            "return": [0.0, 0.01, 0.01, 0.01, 0.01, 0.01],
            "cumulative_return": [0.0, 0.01, 0.0201, 0.0303, 0.0406, 0.0510],
        },
        index=dates,
    )


@pytest.fixture
def sample_metrics() -> dict[str, Any]:
    return {
        "average_return": 0.01,
        "sharpe_ratio": 1.5,
        "max_drawdown": -0.05,
        "monthly_returns": pd.DataFrame(
            [[0.01, 0.02]], index=[2024], columns=["Jan", "Feb"]
        ),
        "yearly_returns": pd.Series([0.12], index=[2024]),
        "weight_history": pd.DataFrame(
            {"AAPL": [0.5], "GOOG": [0.5]},
            index=pd.date_range("2024-01-15", periods=1, freq="ME"),
        ),
    }


# ── Tests ───────────────────────────────────────────────────────────────────


class TestExportAll:
    """Test export_all creates all expected files."""

    def test_creates_all_files(
        self,
        results_df: pd.DataFrame,
        sample_metrics: dict[str, Any],
        tmp_path: Path,
    ) -> None:
        exporter = DataExporter(tmp_path)
        saved = exporter.export_all(results_df, sample_metrics)
        assert (
            len(saved) >= 5
        )  # results csv, metrics csv, metrics json, weight, monthly, yearly
        for path in saved:
            assert Path(path).exists()


class TestResultsCsv:
    """Test results CSV export."""

    def test_creates_csv(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_results_csv(results_df)
        assert len(paths) == 1
        assert Path(paths[0]).name == "backtest_results.csv"

    def test_csv_readable(self, results_df: pd.DataFrame, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_results_csv(results_df)
        loaded = pd.read_csv(paths[0])
        assert len(loaded) == len(results_df)


class TestMetricsExport:
    """Test metrics CSV and JSON exports."""

    def test_metrics_csv_scalar_only(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_metrics_csv(sample_metrics)
        loaded = pd.read_csv(paths[0])
        # Only scalar values should be in the CSV
        assert "average_return" in loaded.columns
        assert "monthly_returns" not in loaded.columns

    def test_metrics_json_valid(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_metrics_json(sample_metrics)
        with open(paths[0]) as f:
            data = json.load(f)
        assert "average_return" in data


class TestWeightHistoryCsv:
    """Test weight history export."""

    def test_exports_when_present(
        self, sample_metrics: dict[str, Any], tmp_path: Path
    ) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_weight_history_csv(sample_metrics)
        assert len(paths) == 1

    def test_skips_when_missing(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        paths = exporter.export_weight_history_csv({})
        assert paths == []


class TestLiveExports:
    """Test live-mode weight and snapshot exports."""

    def test_export_live_weights(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        weights = pd.Series([0.6, 0.4], index=["AAPL", "GOOG"])
        paths = exporter.export_live_weights(weights)
        assert len(paths) == 2  # csv + json
        assert any(p.endswith(".csv") for p in paths)
        assert any(p.endswith(".json") for p in paths)

    def test_export_portfolio_snapshot(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        snapshot = {"cash": 1000.0, "positions": {"AAPL": 10}}
        paths = exporter.export_portfolio_snapshot(snapshot)
        assert len(paths) == 1
        with open(paths[0]) as f:
            data = json.load(f)
        assert data["cash"] == 1000.0


class TestNumpyJsonEncoder:
    """Test that numpy types serialise properly."""

    def test_numpy_types_in_metrics(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        metrics: dict[str, Any] = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14),
            "bool_val": np.bool_(True),
            "array_val": np.array([1.0, 2.0, 3.0]),
        }
        paths = exporter.export_metrics_json(metrics)
        with open(paths[0]) as f:
            data = json.load(f)
        assert data["int_val"] == 42
        assert isinstance(data["float_val"], float)


class TestRebalancingDetailCsv:
    """Test rebalancing detail CSV export."""

    def test_creates_csv(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        detail = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-01-15")],
                "symbol": ["AAPL", "GOOG"],
                "weight": [0.6, 0.4],
                "price": [150.0, 200.0],
                "next_price": [155.0, 205.0],
                "price_change_pct": [3.33, 2.50],
            }
        )
        paths = exporter.export_rebalancing_detail_csv(detail)
        assert len(paths) == 1
        assert Path(paths[0]).name == "rebalancing_detail.csv"

    def test_csv_is_readable_and_correct(self, tmp_path: Path) -> None:
        exporter = DataExporter(tmp_path)
        detail = pd.DataFrame(
            {
                "date": [pd.Timestamp("2024-01-15")],
                "symbol": ["AAPL"],
                "weight": [0.6],
                "price": [150.0],
                "next_price": [160.0],
                "price_change_pct": [6.67],
            }
        )
        paths = exporter.export_rebalancing_detail_csv(detail)
        loaded = pd.read_csv(paths[0])
        assert len(loaded) == 1
        assert loaded["symbol"].iloc[0] == "AAPL"
        assert abs(loaded["weight"].iloc[0] - 0.6) < 0.01
