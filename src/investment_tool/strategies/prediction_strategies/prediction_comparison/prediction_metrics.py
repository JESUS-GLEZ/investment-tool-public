"""Accuracy metrics for prediction strategy comparison.

All metrics operate on aligned (predicted, actual) DataFrames with
rows = evaluation dates and columns = asset symbols.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mean_absolute_error(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Compute mean absolute error across all assets and dates.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames with the same shape.

    Returns
    -------
    float
        MAE (lower is better).
    """
    diff = (predictions - actuals).values.flatten()
    diff = diff[np.isfinite(diff)]
    return float(np.mean(np.abs(diff))) if len(diff) > 0 else float("nan")


def root_mean_squared_error(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Compute root mean squared error across all assets and dates.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.

    Returns
    -------
    float
        RMSE (lower is better).
    """
    diff = (predictions - actuals).values.flatten()
    diff = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(diff**2))) if len(diff) > 0 else float("nan")


def directional_accuracy(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Fraction of predictions that correctly predict the sign of the return.

    A prediction is "directionally correct" if ``sign(predicted) == sign(actual)``.
    Zero predictions/actuals are treated as correct if both are zero.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.

    Returns
    -------
    float
        Directional accuracy in [0, 1] (higher is better).
    """
    pred_flat = predictions.values.flatten()
    act_flat = actuals.values.flatten()
    mask = np.isfinite(pred_flat) & np.isfinite(act_flat)
    pred_flat = pred_flat[mask]
    act_flat = act_flat[mask]
    if len(pred_flat) == 0:
        return float("nan")
    correct = np.sign(pred_flat) == np.sign(act_flat)
    return float(np.mean(correct))


def information_coefficient(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Cross-sectional rank correlation (Spearman IC) averaged over dates.

    At each evaluation date, computes the Spearman rank correlation
    between predicted and realised returns across assets.  The final IC
    is the time-series average.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.

    Returns
    -------
    float
        Mean IC in [-1, 1] (higher is better).
    """
    ics: list[float] = []
    for idx in predictions.index:
        pred_row = predictions.loc[idx].dropna()
        act_row = actuals.loc[idx].dropna()
        common = pred_row.index.intersection(act_row.index)
        if (
            len(common) < 2
            or pred_row[common].nunique() < 2
            or act_row[common].nunique() < 2
        ):
            continue
        corr = pred_row[common].rank().corr(act_row[common].rank())
        if np.isfinite(corr):
            ics.append(float(corr))
    return float(np.mean(ics)) if ics else float("nan")


def mean_signed_error(predictions: pd.DataFrame, actuals: pd.DataFrame) -> float:
    """Compute mean signed error (bias) across all assets and dates.

    Positive bias means the strategy systematically over-predicts.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.

    Returns
    -------
    float
        Mean signed error (closer to 0 is better).
    """
    diff = (predictions - actuals).values.flatten()
    diff = diff[np.isfinite(diff)]
    return float(np.mean(diff)) if len(diff) > 0 else float("nan")


def hit_rate_top_k(
    predictions: pd.DataFrame, actuals: pd.DataFrame, k: int = 5
) -> float:
    """Fraction of dates where the top-k predicted assets overlap with top-k actual.

    Measures whether the strategy correctly identifies the best-performing
    assets, which is what matters most for portfolio construction. Constant
    cross-sections have no ranking signal and are omitted (NaN if all constant).

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.
    k : int, optional
        Number of top assets to compare, by default 5.

    Returns
    -------
    float
        Hit rate in [0, 1] (higher is better).
    """
    hits: list[float] = []
    for idx in predictions.index:
        pred_row = predictions.loc[idx].dropna()
        act_row = actuals.loc[idx].dropna()
        common = pred_row.index.intersection(act_row.index)
        if (
            len(common) < k
            or pred_row[common].nunique() < 2
            or act_row[common].nunique() < 2
        ):
            continue
        top_pred = set(pred_row[common].nlargest(k).index)
        top_act = set(act_row[common].nlargest(k).index)
        overlap = len(top_pred & top_act) / k
        hits.append(overlap)
    return float(np.mean(hits)) if hits else float("nan")


def compute_all_prediction_metrics(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute all accuracy metrics for a prediction strategy.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.
    top_k : int, optional
        Number of assets for hit-rate computation, by default 5.

    Returns
    -------
    dict[str, float]
        Dictionary with keys: ``mae``, ``rmse``, ``directional_accuracy``,
        ``information_coefficient``, ``bias``, ``hit_rate_top_k``.
    """
    return {
        "mae": mean_absolute_error(predictions, actuals),
        "rmse": root_mean_squared_error(predictions, actuals),
        "directional_accuracy": directional_accuracy(predictions, actuals),
        "information_coefficient": information_coefficient(predictions, actuals),
        "bias": mean_signed_error(predictions, actuals),
        "hit_rate_top_k": hit_rate_top_k(predictions, actuals, k=top_k),
    }


def compute_per_asset_metrics(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
) -> pd.DataFrame:
    """Compute accuracy metrics per individual asset.

    Parameters
    ----------
    predictions, actuals : pd.DataFrame
        Aligned DataFrames.

    Returns
    -------
    pd.DataFrame
        One row per symbol with columns: ``mae``, ``rmse``,
        ``directional_accuracy``, ``bias``.
    """
    rows: list[dict[str, float | str]] = []
    for symbol in predictions.columns:
        if symbol not in actuals.columns:
            continue
        pred = predictions[symbol].dropna()
        act = actuals[symbol].dropna()
        common = pred.index.intersection(act.index)
        if len(common) == 0:
            continue
        p = pred[common]
        a = act[common]
        diff = np.asarray((p - a).values)
        p_vals = np.asarray(p.values)
        a_vals = np.asarray(a.values)
        rows.append(
            {
                "symbol": str(symbol),
                "mae": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff**2))),
                "directional_accuracy": float(
                    np.mean(np.sign(p_vals) == np.sign(a_vals))
                ),
                "bias": float(np.mean(diff)),
            }
        )
    return pd.DataFrame(rows).set_index("symbol") if rows else pd.DataFrame()
