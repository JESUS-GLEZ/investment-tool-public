"""Build client-facing figures and metrics from engine results. No job lifecycle."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def present_result(raw_result: dict[str, Any], run_type: str) -> dict[str, Any]:
    """Add ``figures`` and ``comparison_metrics`` to coordinator results.

    The core coordinators save plots to disk but do not return Plotly figure
    dicts.  This function rebuilds figures in-memory using ``PlotBuilder`` so
    the frontend can render them directly without touching the filesystem.

    For comparison run types the ``metrics_df`` DataFrame (rows = metric names,
    columns = strategy/solver/predictor names) is transposed into
    ``comparison_metrics`` (keys = strategy names, values = per-metric dicts)
    which is the shape the frontend table expects.
    """
    try:
        if run_type == "execution":
            raw_result = _add_execution_figures(raw_result)
        elif run_type == "prediction_comparison":
            raw_result = _add_comparison_metrics(raw_result)
            raw_result = _add_prediction_figures(raw_result)
        elif run_type == "solver_comparison":
            raw_result = _add_comparison_metrics(raw_result)
            raw_result = _add_solver_figures(raw_result)
        elif run_type == "strategy_comparison":
            raw_result = _add_comparison_metrics(raw_result)
            raw_result = _add_comparison_figures(raw_result)
    except Exception:
        logger.exception("Failed to enrich result for run_type=%s", run_type)
    return raw_result


def _add_execution_figures(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Build Plotly figures from results_df and inject them as ``figures``."""
    import pandas as pd
    from investment_tool.output.plots import PlotBuilder

    results_df = raw_result.get("results_df")
    metrics = raw_result.get("metrics", {})

    if results_df is None or metrics is None:
        return raw_result

    # results_df may still be a real DataFrame here (pre-sanitize)
    if not isinstance(results_df, pd.DataFrame):
        return raw_result

    try:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            builder = PlotBuilder(
                results_df=results_df,
                metrics=metrics,
                output_dir=Path(tmp),
                save_html=False,
                save_png=False,
            )
            figs = builder.build_all_figures()
            raw_result["figures"] = list(PlotBuilder.figures_to_dict(figs).values())
    except Exception:
        logger.exception("Failed to build execution figures")

    return raw_result


def _add_comparison_metrics(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Transpose metrics_df into comparison_metrics dict for the frontend."""
    import pandas as pd

    metrics_df = raw_result.get("metrics_df")
    if metrics_df is None:
        # solver/prediction comparisons use summary_table
        metrics_df = raw_result.get("summary_table")

    if not isinstance(metrics_df, pd.DataFrame):
        return raw_result

    # metrics_df: rows=metrics, cols=strategies  →  transpose → rows=strategies
    transposed = metrics_df.T  # now rows=strategies, cols=metrics
    comparison_metrics: dict[str, dict[str, Any]] = {}
    for strategy_name, row in transposed.iterrows():
        comparison_metrics[str(strategy_name)] = {
            str(k): (None if (isinstance(v, float) and (v != v)) else v)
            for k, v in row.items()
        }
    raw_result["comparison_metrics"] = comparison_metrics
    return raw_result


def _add_comparison_figures(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Build per-strategy Plotly figures for strategy_comparison runs.

    The strategy coordinator stores individual outputs as::

        individual_results[name] = {"results": list[dict], "portfolio_history": ...}

    ``results`` is the raw per-period snapshot list that
    ``OutputManager._build_results_df`` converts to a DataFrame; we replicate
    that here so we can feed it into ``PlotBuilder``.  Metrics are recomputed
    from the DataFrame since ``compute_additional_metrics`` is not called by
    the coordinator.
    """
    import pandas as pd
    from investment_tool.output.plots import PlotBuilder
    from investment_tool.output.output_manager import OutputManager

    individual = raw_result.get("individual_results")
    if not individual or not isinstance(individual, dict):
        return raw_result

    figures_list: list[dict[str, Any]] = []
    try:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for name, output in individual.items():
                if not isinstance(output, dict):
                    continue

                # Strategy coordinator shape: {"results": list[dict], ...}
                raw_snapshots = output.get("results")
                metrics = output.get("metrics") or {}

                # Older/alternative shape: pre-built DataFrame + metrics
                results_df = output.get("results_df")

                if (
                    raw_snapshots
                    and isinstance(raw_snapshots, list)
                    and not isinstance(results_df, pd.DataFrame)
                ):
                    try:
                        results_df = OutputManager._build_results_df(raw_snapshots)
                    except Exception:
                        logger.debug(
                            "Could not build results_df for strategy %s",
                            name,
                            exc_info=True,
                        )
                        results_df = None

                if not isinstance(results_df, pd.DataFrame):
                    continue

                # New coordinators provide the actual frequency/risk-free rate.
                # Legacy results without frequency still get price/return plots;
                # do not invent an annualization factor for them.
                if not metrics and raw_result.get("frequency"):
                    try:
                        from investment_tool.constants import Frequency
                        from investment_tool.output.metrics import MetricsCalculator

                        rows = (
                            raw_snapshots
                            if isinstance(raw_snapshots, list)
                            else results_df.reset_index()
                            .rename(columns={results_df.index.name or "index": "date"})
                            .to_dict("records")
                        )
                        metrics = MetricsCalculator(
                            rows,
                            output.get("portfolio_history", []),
                            Frequency(raw_result["frequency"]),
                            float(raw_result.get("risk_free_rate", 0.0)),
                        ).compute_all()
                        output["metrics"] = metrics
                    except Exception:
                        logger.warning(
                            "Could not compute metrics for strategy %s",
                            name,
                            exc_info=True,
                        )

                builder = PlotBuilder(
                    results_df=results_df,
                    metrics=metrics,
                    output_dir=Path(tmp) / name,
                    save_html=False,
                    save_png=False,
                )
                figs = builder.build_all_figures()
                for fig_name, fig_dict in PlotBuilder.figures_to_dict(figs).items():
                    layout = fig_dict.get("layout", {})
                    orig_title = layout.get("title", {})
                    if isinstance(orig_title, dict):
                        orig_text = orig_title.get("text", fig_name)
                    else:
                        orig_text = str(orig_title) if orig_title else fig_name
                    layout["title"] = {"text": f"{name} — {orig_text}"}
                    fig_dict["layout"] = layout
                    figures_list.append(fig_dict)
    except Exception:
        logger.exception("Failed to build comparison figures")

    if figures_list:
        raw_result["figures"] = figures_list
    return raw_result


def _add_solver_figures(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Build Plotly figures for solver_comparison runs.

    Reproduces the 3 matplotlib plots from ``comparison_plots.py`` as Plotly
    figures: overlaid Pareto fronts, metric bar charts, and a radar chart.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        results = raw_result.get("results")
        if not results:
            return raw_result

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
        figures_list: list[dict[str, Any]] = []

        # --- 1. Overlaid Pareto Fronts ---
        fig1 = go.Figure()
        for i, r in enumerate(results):
            colour = _COLOURS[i % len(_COLOURS)]
            x = [float(s.objectives[0]) for s in r.front]
            y = [float(-s.objectives[1]) for s in r.front]
            fig1.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(color=colour, size=6, opacity=0.7),
                    name=r.config.label,
                )
            )
        for key, label, colour, mode in [
            (
                "reference_front",
                "MV reference (long-only, no BPRC constraints)",
                "black",
                "lines",
            ),
            ("seed_front", "Multi-Greedy RP seeds", "#d946ef", "markers"),
        ]:
            front = raw_result.get(key) or []
            if front:
                ordered = sorted(front, key=lambda s: s.objectives[0])
                fig1.add_trace(
                    go.Scatter(
                        x=[float(s.objectives[0]) for s in ordered],
                        y=[float(-s.objectives[1]) for s in ordered],
                        name=label,
                        mode=mode,
                        line=dict(color=colour, dash="dash"),
                        marker=dict(color=colour, symbol="x", size=8),
                    )
                )
        fig1.update_layout(
            title_text="Solver Comparison — Pareto Fronts",
            xaxis_title="Portfolio Variance",
            yaxis_title="Estimated Net Return"
            if raw_result.get("trading_costs")
            else "Expected Return",
        )
        figures_list.append(fig1.to_dict())

        # --- 2. Metric bar charts (2×3 grid) ---
        metric_names = [
            "hypervolume",
            "best_sharpe",
            "median_sharpe",
            "spread",
            "cardinality",
            "wall_time",
        ]
        display_names = [
            "Hypervolume ↑",
            "Best Sharpe ↑",
            "Median Sharpe ↑",
            "Spread ↑",
            "Cardinality ↑",
            "Wall Time (s) ↓",
        ]
        labels = [r.config.label for r in results]
        fig2 = make_subplots(rows=2, cols=3, subplot_titles=display_names)
        for idx, (metric, display) in enumerate(zip(metric_names, display_names)):
            row_idx = idx // 3 + 1
            col_idx = idx % 3 + 1
            values = []
            for r in results:
                if metric == "wall_time":
                    values.append(float(r.wall_time_seconds))
                else:
                    values.append(float(r.metrics.get(metric, 0.0)))
            colours = [_COLOURS[i % len(_COLOURS)] for i in range(len(results))]
            fig2.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colours,
                    showlegend=False,
                    text=[f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}" for v in values],
                    textposition="outside",
                ),
                row=row_idx,
                col=col_idx,
            )
        fig2.update_layout(title_text="Solver Comparison — Quality Metrics", height=600)
        figures_list.append(fig2.to_dict())

        # --- 3. Radar chart (normalised, polar) ---
        radar_metrics = [
            "hypervolume",
            "best_sharpe",
            "median_sharpe",
            "spread",
            "cardinality",
        ]
        radar_labels = [
            "Hypervolume",
            "Best Sharpe",
            "Median Sharpe",
            "Spread",
            "Cardinality",
            "Speed",
        ]

        raw_arr = np.array(
            [
                [r.metrics.get(k, 0.0) for k in radar_metrics]
                + [1.0 / max(r.wall_time_seconds, 1e-6)]
                for r in results
            ]
        )
        col_max = raw_arr.max(axis=0)
        col_max[col_max == 0] = 1.0
        normed = raw_arr / col_max

        fig3 = go.Figure()
        for i, r in enumerate(results):
            vals = normed[i].tolist()
            fig3.add_trace(
                go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill="toself",
                    opacity=0.4,
                    name=r.config.label,
                    line=dict(color=_COLOURS[i % len(_COLOURS)]),
                )
            )
        fig3.update_layout(
            title_text="Solver Comparison — Radar",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        )
        figures_list.append(fig3.to_dict())

        if figures_list:
            raw_result["figures"] = figures_list

    except Exception:
        logger.exception("Failed to build solver figures")

    return raw_result


def _add_prediction_figures(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Build Plotly figures from PredictorResult objects for prediction comparisons.

    Reproduces the 5 matplotlib plots from ``prediction_plots.py`` as Plotly
    figure dicts so the frontend can render them without touching the filesystem.
    """
    try:
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        results = raw_result.get("results")
        if not results:
            return raw_result

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
        figures_list: list[dict[str, Any]] = []

        # --- 1. Predicted vs Actual scatter subplots ---
        n = len(results)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        subplot_titles = [r.config.label for r in results]
        fig1 = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
        for i, r in enumerate(results):
            pred = r.predictions_df.values.flatten()
            act = r.actuals_df.values.flatten()
            mask = np.isfinite(pred) & np.isfinite(act)
            pred, act = pred[mask], act[mask]
            colour = _COLOURS[i % len(_COLOURS)]
            row_idx = i // cols + 1
            col_idx = i % cols + 1
            fig1.add_trace(
                go.Scatter(
                    x=act.tolist(),
                    y=pred.tolist(),
                    mode="markers",
                    marker=dict(color=colour, size=4, opacity=0.5),
                    name=r.config.label,
                    showlegend=False,
                ),
                row=row_idx,
                col=col_idx,
            )
            if len(act) > 0:
                lim_min = float(min(act.min(), pred.min()))
                lim_max = float(max(act.max(), pred.max()))
                fig1.add_trace(
                    go.Scatter(
                        x=[lim_min, lim_max],
                        y=[lim_min, lim_max],
                        mode="lines",
                        line=dict(color="black", dash="dash", width=1),
                        showlegend=False,
                    ),
                    row=row_idx,
                    col=col_idx,
                )
        fig1.update_layout(title_text="Predicted vs. Actual Returns", height=400 * rows)
        figures_list.append(fig1.to_dict())

        # --- 2. Error distributions (overlaid histograms) ---
        fig2 = go.Figure()
        for i, r in enumerate(results):
            errors = (r.predictions_df - r.actuals_df).values.flatten()
            errors = errors[np.isfinite(errors)]
            fig2.add_trace(
                go.Histogram(
                    x=errors.tolist(),
                    nbinsx=50,
                    opacity=0.5,
                    name=r.config.label,
                    marker_color=_COLOURS[i % len(_COLOURS)],
                    histnorm="probability density",
                )
            )
        fig2.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
        fig2.update_layout(
            title_text="Prediction Error Distributions",
            barmode="overlay",
            xaxis_title="Prediction Error (predicted − actual)",
            yaxis_title="Density",
        )
        figures_list.append(fig2.to_dict())

        # --- 3. Metric bar charts (2×3 grid) ---
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
        labels = [r.config.label for r in results]
        fig3 = make_subplots(rows=2, cols=3, subplot_titles=display_names)
        for idx, (metric, display) in enumerate(zip(metric_keys, display_names)):
            row_idx = idx // 3 + 1
            col_idx = idx % 3 + 1
            values: list[float | None] = []
            for r in results:
                if metric == "wall_time":
                    values.append(float(r.wall_time_seconds))
                else:
                    val = r.metrics.get(metric, float("nan"))
                    values.append(float(val) if np.isfinite(val) else None)
            colours = [_COLOURS[i % len(_COLOURS)] for i in range(len(results))]
            fig3.add_trace(
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=colours,
                    showlegend=False,
                    text=[
                        "N/A"
                        if v is None
                        else f"{v:.4f}"
                        if abs(v) < 100
                        else f"{v:.1f}"
                        for v in values
                    ],
                    textposition="outside",
                ),
                row=row_idx,
                col=col_idx,
            )
        fig3.update_layout(
            title_text="Prediction Strategy Comparison — Accuracy Metrics", height=600
        )
        figures_list.append(fig3.to_dict())

        # --- 4. Directional accuracy over time (line chart) ---
        fig4 = go.Figure()
        for i, r in enumerate(results):
            pred = r.predictions_df
            act = r.actuals_df
            dir_acc = (np.sign(pred) == np.sign(act)).mean(axis=1)
            fig4.add_trace(
                go.Scatter(
                    x=[str(d) for d in dir_acc.index.tolist()],
                    y=dir_acc.values.tolist(),
                    mode="lines",
                    name=r.config.label,
                    line=dict(color=_COLOURS[i % len(_COLOURS)], width=1.5),
                )
            )
        fig4.add_hline(y=0.5, line_dash="dash", line_color="black", line_width=1)
        fig4.update_layout(
            title_text="Directional Accuracy Over Time",
            xaxis_title="Evaluation Date",
            yaxis_title="Directional Accuracy",
        )
        figures_list.append(fig4.to_dict())

        # --- 5. Per-asset directional accuracy heatmap ---
        for metric in ("directional_accuracy", "mae"):
            all_symbols: list[str] = []
            for r in results:
                if (
                    not r.per_asset_metrics.empty
                    and metric in r.per_asset_metrics.columns
                ):
                    all_symbols.extend(r.per_asset_metrics.index.tolist())
            all_symbols = sorted(set(all_symbols))
            if not all_symbols:
                continue
            strategy_labels = [r.config.label for r in results]
            matrix = []
            for sym in all_symbols:
                row_vals = []
                for r in results:
                    if (
                        not r.per_asset_metrics.empty
                        and metric in r.per_asset_metrics.columns
                        and sym in r.per_asset_metrics.index
                    ):
                        row_vals.append(float(r.per_asset_metrics.loc[sym, metric]))
                    else:
                        row_vals.append(float("nan"))
                matrix.append(row_vals)
            fig5 = go.Figure(
                go.Heatmap(
                    z=matrix,
                    x=strategy_labels,
                    y=all_symbols,
                    colorscale="RdYlGn",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title=metric),
                )
            )
            fig5.update_layout(
                title_text=f"Per-Asset {metric.replace('_', ' ').title()}",
                height=max(400, len(all_symbols) * 14),
            )
            figures_list.append(fig5.to_dict())
            break  # only one heatmap needed; directional_accuracy preferred

        if figures_list:
            raw_result["figures"] = figures_list

    except Exception:
        logger.exception("Failed to build prediction figures")

    return raw_result
