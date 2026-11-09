"""
Utility functions for the Investment Tool.

This module provides common helper functions used throughout the application,
including file operations, time calculations, timestamp alignment, and
covariance matrix computations for portfolio optimization.
"""

from __future__ import annotations

import datetime
import logging
import os

import numpy as np
import pandas as pd

from investment_tool.defaults import COVARIANCE, PREDICTION

logger = logging.getLogger(__name__)


def get_extension(file: str) -> str:
    """Extract the file extension from a filename.

    Parameters
    ----------
    file : str
        Filename or path to extract extension from.

    Returns
    -------
    str
        File extension including the dot (e.g., ".csv", ".pkl").
    """
    _, extension = os.path.splitext(file)
    return extension


def time_diff(start_time: datetime.datetime, end_time: datetime.datetime) -> int:
    """Calculate the difference in minutes between two datetime objects.

    Parameters
    ----------
    start_time : datetime.datetime
        Start time.
    end_time : datetime.datetime
        End time.

    Returns
    -------
    int
        Difference in minutes, rounded to nearest integer.
    """
    return round((end_time - start_time).total_seconds() / 60)


def align_timestamps(data: pd.DataFrame, offset_hours: int = 0) -> pd.DataFrame:
    """Adjust DataFrame timestamps by a given number of hours.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with a DatetimeIndex.
    offset_hours : int, optional
        Hours to subtract from timestamps, by default 0.

    Returns
    -------
    pd.DataFrame
        DataFrame with adjusted timestamps (modified in place).
    """
    data.index = data.index - pd.Timedelta(hours=offset_hours)  # type: ignore[operator]
    return data


def recently_observed_symbols(
    returns: pd.DataFrame,
    min_periods: int = PREDICTION.min_regression_periods,
) -> list[str]:
    """Keep assets with a complete recent return window.

    A newly listed or stale asset must not erase the shared training window
    when multi-asset predictors and covariance estimators join their series.
    For short test/live histories, use all but the first (return-less) bar.
    """
    if len(returns) < 3:
        return []
    window = min(min_periods, len(returns) - 1)
    recent = returns.sort_index().replace([np.inf, -np.inf], np.nan).tail(window)
    return [
        str(symbol)
        for symbol in returns.columns
        if recent[symbol].notna().all() and recent[symbol].nunique() > 1
    ]


def _final_ewm_correlation(observed: pd.DataFrame, span: int) -> np.ndarray:
    """Final pairwise EWM correlation, with O(assets²) working memory.

    Online centered moments follow pandas' adjust=True, ignore_na=False
    recurrence. A missing member excludes the observation from that pair,
    while its old weight still decays. Pair-specific means and variances are
    essential: marginal variances cannot normalize pairwise correlations.
    Only the final moments are retained, not a matrix for every timestamp.
    """
    n = len(observed.columns)
    weight = np.zeros((n, n))
    mean = np.zeros((n, n))
    variance = np.zeros((n, n))
    covariance = np.zeros((n, n))
    count = np.zeros((n, n), dtype=np.int64)
    decay = 1.0 - 2.0 / (span + 1.0)
    for row in observed.to_numpy(dtype=float):
        valid = np.isfinite(row)
        paired = valid[:, None] & valid[None, :]
        value = np.where(valid, row, 0.0)[:, None]
        weight *= decay
        denominator = weight + 1.0
        # Preserve exact constants, as pandas does, to avoid spurious variance.
        updated_mean = np.where(
            paired & (mean != value), (weight * mean + value) / denominator, mean
        )
        old_delta = mean - updated_mean
        new_delta = value - updated_mean
        variance = np.where(
            paired,
            (weight * (variance + old_delta**2) + new_delta**2) / denominator,
            variance,
        )
        covariance = np.where(
            paired,
            (weight * (covariance + old_delta * old_delta.T) + new_delta * new_delta.T)
            / denominator,
            covariance,
        )
        mean = updated_mean
        weight += paired
        count += paired
    # The common unbiased weight correction cancels in the correlation ratio.
    denominator = np.sqrt(variance * variance.T)
    correlation = np.full((n, n), np.nan)
    np.divide(
        covariance, denominator, out=correlation, where=(count >= 2) & (denominator > 0)
    )
    return correlation


def build_exp_weighted_covariance_matrix(
    returns: pd.DataFrame,
    span: int = COVARIANCE.ewm_span,
    clip_threshold: float | None = None,
    shrinkage: float = COVARIANCE.shrinkage,
) -> pd.DataFrame:
    """Estimate covariance from observed returns, then regularize correlations.

    Each marginal EWM variance uses its asset's complete available history.
    Pairwise EWM correlations use overlapping observations, without filling
    missing returns or discarding rows because another asset was not listed.
    The caller must pass only returns observable at the decision cutoff.

    Pairwise estimates need not form a PSD matrix. Repair the correlation
    matrix, normalize its diagonal, then shrink it toward the identity by
    ``shrinkage`` (default 0.1). This preserves marginal variances and gives
    the final correlation matrix an eigenvalue floor equal to ``shrinkage``.
    It is diagonal shrinkage, not the Ledoit-Wolf estimator, and is not fitted
    to backtest profit. ``span`` counts observation periods, not calendar days.

    Raise on insufficient overlap or unestimable variances rather than assign
    made-up risk. Diagnostics are retained in the returned frame's attrs.
    """
    if not np.isfinite(span) or span <= 1:
        raise ValueError("Covariance span must be finite and greater than one")
    if not np.isfinite(shrinkage) or not 0 < shrinkage <= 1:
        raise ValueError("Covariance shrinkage must be in (0, 1]")
    if clip_threshold is not None and (
        not np.isfinite(clip_threshold) or clip_threshold <= 0
    ):
        raise ValueError("Return clip threshold must be finite and positive")
    if returns.empty or not returns.columns.is_unique or not returns.index.is_unique:
        raise ValueError("Covariance returns must be nonempty with unique labels")

    observed = returns.sort_index().astype(float).replace([np.inf, -np.inf], np.nan)
    if clip_threshold is not None:
        observed = observed.clip(-clip_threshold, clip_threshold)
    valid = observed.notna().to_numpy(dtype=np.int64)
    overlap = valid.T @ valid
    if np.any(overlap < 2):
        raise ValueError("Covariance requires at least two observed returns per pair")

    window = observed.ewm(span=span, min_periods=2, adjust=True, ignore_na=False)
    variances = window.var().iloc[-1].to_numpy(dtype=float)
    if not np.isfinite(variances).all() or np.any(variances <= 0):
        invalid = observed.columns[~np.isfinite(variances) | (variances <= 0)]
        raise ValueError(f"Unestimable return variance for: {list(invalid)}")

    correlation = _final_ewm_correlation(observed, span)
    if not np.isfinite(correlation).all():
        invalid_pairs = np.argwhere(~np.isfinite(correlation))
        examples = [
            (str(observed.columns[i]), str(observed.columns[j]))
            for i, j in invalid_pairs[:5]
        ]
        raise ValueError(f"Unestimable pairwise return correlation: {examples}")
    correlation = np.clip((correlation + correlation.T) / 2, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    # Clip only statistical inconsistency here; shrinkage supplies the risk
    # floor. Renormalization preserves the independent marginal estimates.
    repaired = (eigenvectors * np.maximum(eigenvalues, 0)) @ eigenvectors.T
    diagonal = np.sqrt(np.diag(repaired))
    repaired /= np.outer(diagonal, diagonal)
    regularized = (1 - shrinkage) * repaired + shrinkage * np.eye(len(variances))
    scales = np.sqrt(variances)
    covariance = regularized * np.outer(scales, scales)
    covariance = (covariance + covariance.T) / 2
    result = pd.DataFrame(covariance, index=observed.columns, columns=observed.columns)

    regularized_eigenvalues = np.linalg.eigvalsh(regularized)
    diagnostics = {
        "method": "pairwise_ewm_diagonal_shrinkage_v1",
        "span": span,
        "shrinkage": shrinkage,
        "observed_returns": {
            str(symbol): int(count)
            for symbol, count in zip(observed.columns, np.diag(overlap), strict=True)
        },
        "minimum_pair_overlap": int(overlap.min()),
        "raw_correlation_min_eigenvalue": float(eigenvalues.min()),
        "regularized_correlation_min_eigenvalue": float(regularized_eigenvalues.min()),
        "regularized_correlation_condition_number": float(
            regularized_eigenvalues.max() / regularized_eigenvalues.min()
        ),
    }
    result.attrs["covariance_diagnostics"] = diagnostics
    logger.info(
        "Covariance: %d assets, span=%s, shrinkage=%s, min overlap=%d, "
        "correlation condition=%.2f",
        len(variances),
        span,
        shrinkage,
        overlap.min(),
        diagnostics["regularized_correlation_condition_number"],
    )
    return result


def _symmetrise(cov: pd.DataFrame) -> pd.DataFrame:
    """Force a DataFrame covariance matrix to be exactly symmetric.

    Parameters
    ----------
    cov : pd.DataFrame
        Approximately symmetric covariance matrix.

    Returns
    -------
    pd.DataFrame
        Exactly symmetric covariance matrix.
    """
    return (cov + cov.T) / 2.0


def validate_covariance_matrix(cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a covariance matrix.

    Replaces NaN/Inf values: diagonal entries become 0.01 (small positive),
    off-diagonal entries become 0.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Covariance matrix to validate.

    Returns
    -------
    pd.DataFrame
        Cleaned covariance matrix with no NaN or Inf values.
    """
    for i in range(cov_matrix.shape[0]):
        for j in range(cov_matrix.shape[1]):
            val = cov_matrix.iloc[i, j]
            if np.isnan(val) or np.isinf(val):  # type: ignore[arg-type]
                cov_matrix.iloc[i, j] = 0.01 if i == j else 0
    return cov_matrix


def nearest_psd(cov_matrix: pd.DataFrame, tol: float = 1e-10) -> pd.DataFrame:
    """Compute the nearest positive semi-definite matrix using Higham's algorithm.

    Parameters
    ----------
    cov_matrix : pd.DataFrame
        Original covariance matrix (may have negative eigenvalues).
    tol : float, optional
        Minimum eigenvalue threshold, by default 1e-10.

    Returns
    -------
    pd.DataFrame
        PSD-adjusted covariance matrix with same index/columns.
    """
    A = cov_matrix.to_numpy()

    # Symmetrize
    A = 0.5 * (A + A.T)

    # Eigen-decompose
    eigvals, eigvecs = np.linalg.eigh(A)

    # Clip negative eigenvalues
    eigvals_clipped = np.clip(eigvals, tol, None)

    # Reconstruct
    A_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Resymmetrize (handle numerical precision)
    A_psd = 0.5 * (A_psd + A_psd.T)

    return pd.DataFrame(A_psd, index=cov_matrix.index, columns=cov_matrix.columns)
