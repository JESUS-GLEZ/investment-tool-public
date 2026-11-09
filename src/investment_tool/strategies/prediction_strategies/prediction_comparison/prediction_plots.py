"""Comparison plots for prediction strategy benchmarking.

Generates prediction-vs-actual scatter plots, error distributions,
metric bar charts, and per-asset heatmaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_result import (
        PredictorResult,
    )

logger = logging.getLogger(__name__)

_COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _resolve_path(output_dir: Path | None, name: str) -> Path:
    """Build save path under ``<output_dir>/prediction_comparison/``."""
    base = (
        Path(output_dir) / "prediction_comparison"
        if output_dir
        else Path.cwd() / "prediction_comparison"
    )
    base.mkdir(parents=True, exist_ok=True)
    return base / name


def plot_predictions_vs_actuals(
    results: list[PredictorResult],
    output_directory: Path | None = None,
) -> str | None:
    """Scatter plot of predicted vs. realised returns per strategy.

    One subplot per predictor, with the 45° ideal line.

    Parameters
    ----------
    results : list[PredictorResult]
        Benchmark results.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` if no results.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    n = len(results)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    save_path = _resolve_path(output_directory, "predictions_vs_actuals.png")

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for i, r in enumerate(results):
        ax = axes[i // cols][i % cols]
        pred = r.predictions_df.values.flatten()
        act = r.actuals_df.values.flatten()
        mask = np.isfinite(pred) & np.isfinite(act)
        pred, act = pred[mask], act[mask]

        colour = _COLOURS[i % len(_COLOURS)]
        ax.scatter(act, pred, alpha=0.4, s=12, c=colour)

        # 45° line
        lims = [
            min(act.min(), pred.min()) if len(act) > 0 else -0.1,
            max(act.max(), pred.max()) if len(act) > 0 else 0.1,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Actual Return")
        ax.set_ylabel("Predicted Return")
        ax.set_title(r.config.label, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Predicted vs. Actual Returns", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved predictions vs. actuals plot: %s", save_path)
    return str(save_path)


def plot_error_distributions(
    results: list[PredictorResult],
    output_directory: Path | None = None,
) -> str | None:
    """Overlaid histogram of prediction errors per strategy.

    Parameters
    ----------
    results : list[PredictorResult]
        Benchmark results.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` if no results.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "error_distributions.png")
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, r in enumerate(results):
        errors = (r.predictions_df - r.actuals_df).values.flatten()
        errors = errors[np.isfinite(errors)]
        colour = _COLOURS[i % len(_COLOURS)]
        ax.hist(
            errors,
            bins=50,
            alpha=0.5,
            label=r.config.label,
            color=colour,
            density=True,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction Error (predicted − actual)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Error Distributions")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved error distributions plot: %s", save_path)
    return str(save_path)


def plot_metric_bars(
    results: list[PredictorResult],
    output_directory: Path | None = None,
) -> str | None:
    """Side-by-side bar chart comparing all prediction accuracy metrics.

    Parameters
    ----------
    results : list[PredictorResult]
        Benchmark results.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` if no results.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "prediction_metric_bars.png")

    labels = [r.config.label for r in results]
    metric_keys = [
        "mae",
        "rmse",
        "directional_accuracy",
        "information_coefficient",
        "hit_rate_top_k",
        "wall_time",
    ]
    display_names = [
        "MAE ↓",
        "RMSE ↓",
        "Dir. Accuracy ↑",
        "Info. Coeff. ↑",
        "Hit Rate Top-K ↑",
        "Wall Time (s) ↓",
    ]

    n_solvers = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes_flat = axes.flatten()

    for idx, (metric, display) in enumerate(zip(metric_keys, display_names)):
        ax = axes_flat[idx]
        values = []
        for r in results:
            if metric == "wall_time":
                values.append(r.wall_time_seconds)
            else:
                val = r.metrics.get(metric, float("nan"))
                values.append(val if np.isfinite(val) else float("nan"))

        colours = [_COLOURS[i % len(_COLOURS)] for i in range(n_solvers)]
        bars = ax.bar(labels, values, color=colours, alpha=0.8)
        ax.set_title(display, fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8, rotation=30)

        for bar, val in zip(bars, values):
            if not np.isfinite(val):
                continue
            fmt = f"{val:.4f}" if abs(val) < 100 else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt,
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.suptitle(
        "Prediction Strategy Comparison — Accuracy Metrics",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved prediction metric bars plot: %s", save_path)
    return str(save_path)


def plot_directional_accuracy_over_time(
    results: list[PredictorResult],
    output_directory: Path | None = None,
) -> str | None:
    """Rolling directional accuracy over evaluation dates.

    Parameters
    ----------
    results : list[PredictorResult]
        Benchmark results.
    output_directory : Path | None
        Output directory for the plot.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` if no results.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, "directional_accuracy_over_time.png")
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, r in enumerate(results):
        pred = r.predictions_df
        act = r.actuals_df
        dir_acc_per_date = (np.sign(pred) == np.sign(act)).mean(axis=1)
        colour = _COLOURS[i % len(_COLOURS)]
        ax.plot(
            dir_acc_per_date.index,
            dir_acc_per_date.values,
            label=r.config.label,
            color=colour,
            linewidth=1.5,
            alpha=0.8,
        )

    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Evaluation Date")
    ax.set_ylabel("Directional Accuracy")
    ax.set_title("Directional Accuracy Over Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved directional accuracy over time plot: %s", save_path)
    return str(save_path)


def plot_per_asset_heatmap(
    results: list[PredictorResult],
    output_directory: Path | None = None,
    metric: str = "directional_accuracy",
) -> str | None:
    """Heatmap of a per-asset metric across prediction strategies.

    Parameters
    ----------
    results : list[PredictorResult]
        Benchmark results.
    output_directory : Path | None
        Output directory for the plot.
    metric : str, optional
        Which per-asset metric to plot, by default ``"directional_accuracy"``.

    Returns
    -------
    str | None
        Path to saved plot, or ``None`` if no results.
    """
    import matplotlib.pyplot as plt

    if not results:
        return None

    save_path = _resolve_path(output_directory, f"per_asset_{metric}_heatmap.png")

    # Build matrix: rows=assets, columns=strategies
    all_symbols: list[str] = []
    for r in results:
        if not r.per_asset_metrics.empty and metric in r.per_asset_metrics.columns:
            all_symbols.extend(r.per_asset_metrics.index.tolist())
    all_symbols = sorted(set(all_symbols))

    if not all_symbols:
        return None

    matrix = np.full((len(all_symbols), len(results)), np.nan)
    labels = [r.config.label for r in results]

    for j, r in enumerate(results):
        if r.per_asset_metrics.empty or metric not in r.per_asset_metrics.columns:
            continue
        for i, sym in enumerate(all_symbols):
            if sym in r.per_asset_metrics.index:
                matrix[i, j] = r.per_asset_metrics.loc[sym, metric]

    fig, ax = plt.subplots(
        figsize=(max(6, len(results) * 1.5), max(6, len(all_symbols) * 0.35))
    )
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30)
    ax.set_yticks(range(len(all_symbols)))
    ax.set_yticklabels(all_symbols, fontsize=7)
    ax.set_title(f"Per-Asset {metric.replace('_', ' ').title()}", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300)
    plt.close(fig)
    logger.info("Saved per-asset heatmap: %s", save_path)
    return str(save_path)
