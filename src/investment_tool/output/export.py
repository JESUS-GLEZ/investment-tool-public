"""
Structured data export for investment analysis results.

This module provides the DataExporter class for saving backtest/live results
and computed metrics to CSV and JSON formats with consistent formatting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder with numpy/pandas type support.

    Handles ``np.integer``, ``np.floating``, ``np.bool_``, ``np.ndarray``,
    ``pd.Timestamp``, and ``pd.Series`` serialisation.
    """

    def default(self, o: Any) -> Any:
        """Encode numpy/pandas types to JSON-compatible Python types.

        Parameters
        ----------
        o : Any
            Object to encode.

        Returns
        -------
        Any
            JSON-compatible representation.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (pd.Timestamp, pd.Period)):
            return str(o)
        if isinstance(o, pd.Series):
            return {str(k): v for k, v in o.to_dict().items()}
        if isinstance(o, pd.DataFrame):
            # Convert index to strings to avoid non-string dict keys
            o = o.copy()
            o.index = o.index.map(str)
            return o.to_dict(orient="list")
        # Fallback: convert to string rather than raising TypeError
        try:
            return super().default(o)
        except TypeError:
            return str(o)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-string dict keys to strings.

    JSON requires all dict keys to be strings.  This function walks
    nested structures and converts any non-str keys (e.g. pd.Timestamp,
    np.int64) to their string representations.

    Parameters
    ----------
    obj : Any
        Object to sanitize.

    Returns
    -------
    Any
        Sanitized copy safe for ``json.dump``.
    """
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)) and not np.isfinite(obj):
        return None
    if isinstance(obj, pd.Series):
        return _sanitize_for_json(obj.to_dict())
    if isinstance(obj, pd.DataFrame):
        return _sanitize_for_json(obj.to_dict(orient="list"))
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


class DataExporter:
    """Export investment results and metrics to CSV and JSON files.

    Parameters
    ----------
    output_dir : Path
        Directory where export files will be saved.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        results_df: pd.DataFrame,
        metrics: dict[str, Any],
        rebalancing_detail: pd.DataFrame | None = None,
    ) -> list[str]:
        """Export all data artifacts (CSVs and JSONs).

        Parameters
        ----------
        results_df : pd.DataFrame
            Date-indexed backtest results.
        metrics : dict[str, Any]
            Computed metrics dictionary.
        rebalancing_detail : pd.DataFrame | None, optional
            Per-asset weights and prices at each rebalancing period.

        Returns
        -------
        list[str]
            Paths of saved files.
        """
        saved: list[str] = []
        saved.extend(self.export_results_csv(results_df))
        saved.extend(self.export_metrics_csv(metrics))
        saved.extend(self.export_metrics_json(metrics))
        saved.extend(self.export_weight_history_csv(metrics))
        saved.extend(self.export_monthly_returns_csv(metrics))
        saved.extend(self.export_yearly_returns_csv(metrics))
        if rebalancing_detail is not None and not rebalancing_detail.empty:
            saved.extend(self.export_rebalancing_detail_csv(rebalancing_detail))
        return saved

    def export_results_csv(self, results_df: pd.DataFrame) -> list[str]:
        """Save backtest time-series results to CSV.

        Parameters
        ----------
        results_df : pd.DataFrame
            Date-indexed results with total_value, return, cumulative_return.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        path = self._output_dir / "backtest_results.csv"
        results_df.to_csv(path)
        logger.debug(f"Exported results CSV: {path}")
        return [str(path)]

    def export_rebalancing_detail_csv(
        self, rebalancing_detail: pd.DataFrame
    ) -> list[str]:
        """Save per-asset rebalancing detail to CSV.

        Each row contains a selected asset's weight, its price at the
        rebalancing period, and its price at the next period.

        Parameters
        ----------
        rebalancing_detail : pd.DataFrame
            Columns: date, symbol, weight, price, next_price,
            price_change_pct.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        path = self._output_dir / "rebalancing_detail.csv"
        rebalancing_detail.to_csv(path, index=False)
        logger.debug(f"Exported rebalancing detail CSV: {path}")
        return [str(path)]

    def export_metrics_csv(self, metrics: dict[str, Any]) -> list[str]:
        """Save scalar metrics to a single-row CSV.

        Parameters
        ----------
        metrics : dict[str, Any]
            Metrics dictionary; only scalar values are exported.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        scalar = {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float, str, np.integer, np.floating))
        }
        path = self._output_dir / "backtest_metrics.csv"
        pd.DataFrame([scalar]).to_csv(path, index=False)
        logger.debug(f"Exported metrics CSV: {path}")
        return [str(path)]

    def export_metrics_json(self, metrics: dict[str, Any]) -> list[str]:
        """Save full metrics (including series) to JSON.

        Non-serialisable objects (DataFrames, Series) are converted
        automatically via ``_NumpyJSONEncoder``.

        Parameters
        ----------
        metrics : dict[str, Any]
            Full metrics dictionary.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        path = self._output_dir / "backtest_metrics.json"
        sanitized = _sanitize_for_json(metrics)
        with path.open("w", encoding="utf-8") as f:
            json.dump(sanitized, f, cls=_NumpyJSONEncoder, indent=2, allow_nan=False)
        logger.debug(f"Exported metrics JSON: {path}")
        return [str(path)]

    def export_weight_history_csv(self, metrics: dict[str, Any]) -> list[str]:
        """Save weight history DataFrame to CSV.

        Parameters
        ----------
        metrics : dict[str, Any]
            Must contain ``weight_history`` key (pd.DataFrame).

        Returns
        -------
        list[str]
            Saved file paths (empty if no weight history).
        """
        wh: pd.DataFrame | None = metrics.get("weight_history")
        if wh is None or (isinstance(wh, pd.DataFrame) and wh.empty):
            return []
        path = self._output_dir / "weight_history.csv"
        wh.to_csv(path)
        logger.debug(f"Exported weight history CSV: {path}")
        return [str(path)]

    def export_monthly_returns_csv(self, metrics: dict[str, Any]) -> list[str]:
        """Save monthly returns matrix to CSV.

        Parameters
        ----------
        metrics : dict[str, Any]
            Must contain ``monthly_returns`` key (pd.DataFrame).

        Returns
        -------
        list[str]
            Saved file paths (empty if no monthly returns).
        """
        mr: pd.DataFrame | None = metrics.get("monthly_returns")
        if mr is None or (isinstance(mr, pd.DataFrame) and mr.empty):
            return []
        path = self._output_dir / "monthly_returns.csv"
        mr.to_csv(path)
        logger.debug(f"Exported monthly returns CSV: {path}")
        return [str(path)]

    def export_yearly_returns_csv(self, metrics: dict[str, Any]) -> list[str]:
        """Save yearly returns series to CSV.

        Parameters
        ----------
        metrics : dict[str, Any]
            Must contain ``yearly_returns`` key (pd.Series).

        Returns
        -------
        list[str]
            Saved file paths (empty if no yearly returns).
        """
        yr: pd.Series | None = metrics.get("yearly_returns")
        if yr is None or (isinstance(yr, pd.Series) and yr.empty):
            return []
        path = self._output_dir / "yearly_returns.csv"
        yr.to_csv(path, header=True)
        logger.debug(f"Exported yearly returns CSV: {path}")
        return [str(path)]

    # ------------------------------------------------------------------
    # Live-mode helpers
    # ------------------------------------------------------------------

    def export_live_weights(self, weights: pd.Series) -> list[str]:
        """Save optimal weights for live investment mode.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by asset symbol.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        saved: list[str] = []

        csv_path = self._output_dir / "optimal_weights.csv"
        weights.to_frame(name="weight").to_csv(csv_path)
        saved.append(str(csv_path))

        json_path = self._output_dir / "optimal_weights.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(weights.to_dict(), f, cls=_NumpyJSONEncoder, indent=2)
        saved.append(str(json_path))

        logger.debug(f"Exported live weights to {self._output_dir}")
        return saved

    def export_portfolio_snapshot(self, snapshot: dict[str, Any]) -> list[str]:
        """Save portfolio snapshot for live investment mode.

        Parameters
        ----------
        snapshot : dict[str, Any]
            Portfolio state dictionary.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        path = self._output_dir / "portfolio_snapshot.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, cls=_NumpyJSONEncoder, indent=2, default=str)
        logger.debug(f"Exported portfolio snapshot: {path}")
        return [str(path)]
