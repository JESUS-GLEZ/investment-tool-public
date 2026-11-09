"""Unit tests for prediction comparison plots.

Verifies that plot functions produce files without errors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_metrics import (
    compute_all_prediction_metrics,
    compute_per_asset_metrics,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_plots import (
    plot_directional_accuracy_over_time,
    plot_error_distributions,
    plot_metric_bars,
    plot_per_asset_heatmap,
    plot_predictions_vs_actuals,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_config import (
    PredictorConfig,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_result import (
    PredictorResult,
)
from investment_tool.strategies.prediction_strategies.average_prediction_strategy import (
    AveragePredictionStrategy,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result(label: str, n_dates: int = 10, n_assets: int = 4) -> PredictorResult:
    np.random.seed(hash(label) % 2**31)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="ME")
    symbols = [f"S{i}" for i in range(n_assets)]
    actuals = pd.DataFrame(
        np.random.randn(n_dates, n_assets) * 0.05,
        index=dates,
        columns=symbols,
    )
    predictions = actuals + np.random.randn(n_dates, n_assets) * 0.01

    return PredictorResult(
        config=PredictorConfig(
            label=label,
            factory=lambda s, ss: AveragePredictionStrategy(s),
        ),
        predictions_df=predictions,
        actuals_df=actuals,
        metrics=compute_all_prediction_metrics(predictions, actuals),
        per_asset_metrics=compute_per_asset_metrics(predictions, actuals),
        wall_time_seconds=1.0,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestPlotPredVsActuals:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        path = plot_predictions_vs_actuals(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty(self) -> None:
        assert plot_predictions_vs_actuals([]) is None


class TestPlotErrorDistributions:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("X"), _make_result("Y")]
        path = plot_error_distributions(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty(self) -> None:
        assert plot_error_distributions([]) is None


class TestPlotMetricBars:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("M1"), _make_result("M2")]
        path = plot_metric_bars(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty(self) -> None:
        assert plot_metric_bars([]) is None


class TestPlotDirAccOverTime:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("T1"), _make_result("T2")]
        path = plot_directional_accuracy_over_time(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty(self) -> None:
        assert plot_directional_accuracy_over_time([]) is None


class TestPlotPerAssetHeatmap:
    def test_creates_file(self, tmp_path: Path) -> None:
        results = [_make_result("H1"), _make_result("H2")]
        path = plot_per_asset_heatmap(results, output_directory=tmp_path)
        assert path is not None
        assert Path(path).exists()

    def test_empty(self) -> None:
        assert plot_per_asset_heatmap([]) is None
