"""Predictor result container.

Holds per-step predictions, realised returns, accuracy metrics, and
timing for a single prediction strategy evaluated via walk-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_config import (
    PredictorConfig,
)


@dataclass
class PredictorResult:
    """Output of evaluating one prediction strategy via walk-forward.

    Parameters
    ----------
    config : PredictorConfig
        The predictor configuration that produced this result.
    predictions_df : pd.DataFrame
        DataFrame with rows = target-period labels, columns = symbols.
        Values are predicted returns.
    actuals_df : pd.DataFrame
        DataFrame with rows = target-period labels, columns = symbols.
        Values are realised returns.
    metrics : dict[str, float]
        Aggregate accuracy metrics (MAE, RMSE, directional accuracy, etc.).
    per_asset_metrics : pd.DataFrame
        Per-asset accuracy metrics (one row per symbol).
    wall_time_seconds : float
        Total wall-clock time for all walk-forward steps.
    """

    config: PredictorConfig
    predictions_df: pd.DataFrame
    actuals_df: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    per_asset_metrics: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    wall_time_seconds: float = 0.0
    evaluation_dates: list[str] = field(default_factory=list)
    decision_times: list[str] = field(default_factory=list)
    failed_periods: dict[str, str] = field(default_factory=dict)
