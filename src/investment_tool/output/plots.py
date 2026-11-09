"""
Professional interactive and static chart generation for investment analysis.

This module provides the PlotBuilder class which creates a comprehensive suite
of charts from backtest/live results and metrics using Plotly for interactive
HTML output and optional static PNG fallback.

Charts include equity curves, drawdown series, rolling Sharpe, return
distributions, monthly heatmaps, weight evolution, turnover, and
portfolio concentration analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Colour palette ──────────────────────────────────────────────────────────
_COLOURS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "negative": "#d62728",
    "positive": "#2ca02c",
    "neutral": "#7f7f7f",
    "drawdown_fill": "rgba(214,39,40,0.25)",
    "grid": "rgba(200,200,200,0.4)",
}


class PlotBuilder:
    """Interactive Plotly chart builder for investment analysis.

    Generates a full suite of HTML (interactive) and PNG (static) charts
    from backtest results and computed metrics.

    Parameters
    ----------
    results_df : pd.DataFrame
        Date-indexed DataFrame with at least ``total_value`` and ``return``
        columns (output of ``Backtester.get_results()``).
    metrics : dict[str, Any]
        Metrics dictionary returned by ``MetricsCalculator.compute_all()``.
    output_dir : Path
        Directory where plots will be saved.
    save_html : bool, optional
        Whether to save interactive HTML versions, by default True.
    save_png : bool, optional
        Whether to save static PNG versions, by default True.
    """

    def __init__(
        self,
        results_df: pd.DataFrame,
        metrics: dict[str, Any],
        output_dir: Path,
        save_html: bool = True,
        save_png: bool = True,
    ) -> None:
        self._df = results_df.copy()
        self._metrics = metrics
        self._output_dir = output_dir
        self._save_html = save_html
        self._save_png = save_png

        self._plots_dir = self._output_dir / "plots"
        self._plots_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def build_all(self) -> list[str]:
        """Generate all available plots.

        Returns
        -------
        list[str]
            List of file paths for all saved plot files.
        """
        logger.info("Generating full plot suite")
        saved: list[str] = []

        builders = [
            self.plot_equity_curve,
            self.plot_drawdown,
            self.plot_rolling_sharpe,
            self.plot_return_distribution,
            self.plot_monthly_heatmap,
            self.plot_cumulative_return,
            self.plot_weight_evolution,
            self.plot_turnover,
            self.plot_underwater,
        ]

        for builder in builders:
            try:
                paths = builder()
                saved.extend(paths)
            except Exception:
                logger.exception(f"Failed to generate plot: {builder.__name__}")

        logger.info(f"Generated {len(saved)} plot files")
        return saved

    def build_all_figures(self) -> dict[str, go.Figure]:
        """Build all plot figures without saving them to disk.

        This is the primary entry point for the web API — the returned
        figures can be serialised to JSON via :meth:`figure_to_json` or
        :meth:`figure_to_dict` and sent directly to a React/Plotly
        frontend.

        Returns
        -------
        dict[str, go.Figure]
            Mapping of plot name to Plotly Figure object.
            Only plots whose data prerequisites are satisfied are
            included (e.g. no ``drawdown`` if ``drawdown_series`` is
            missing from metrics).
        """
        builders: dict[str, Any] = {
            "equity_curve": self._build_equity_curve,
            "drawdown": self._build_drawdown,
            "rolling_sharpe": self._build_rolling_sharpe,
            "return_distribution": self._build_return_distribution,
            "monthly_heatmap": self._build_monthly_heatmap,
            "cumulative_return": self._build_cumulative_return,
            "weight_evolution": self._build_weight_evolution,
            "turnover": self._build_turnover,
            "underwater": self._build_underwater,
        }

        figures: dict[str, go.Figure] = {}
        for name, builder in builders.items():
            try:
                fig = builder()
                if fig is not None:
                    figures[name] = fig
            except Exception:
                logger.exception("Failed to build figure: %s", name)

        logger.info("Built %d plot figures", len(figures))
        return figures

    @staticmethod
    def figure_to_dict(fig: go.Figure) -> dict[str, Any]:
        """Convert a Plotly Figure to a plain dict suitable for JSON.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure.

        Returns
        -------
        dict[str, Any]
            Dictionary with ``data`` and ``layout`` keys matching the
            Plotly JSON schema expected by ``react-plotly.js``.
        """
        return fig.to_plotly_json()  # type: ignore[no-any-return]

    @staticmethod
    def figure_to_json(fig: go.Figure) -> str:
        """Serialise a Plotly Figure to a JSON string.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure.

        Returns
        -------
        str
            JSON string consumable by ``Plotly.react()`` or
            ``react-plotly.js``.
        """
        return json.dumps(fig.to_plotly_json())

    @staticmethod
    def figures_to_dict(
        figures: dict[str, go.Figure],
    ) -> dict[str, dict[str, Any]]:
        """Batch-convert named figures to JSON-serialisable dicts.

        Convenience wrapper around :meth:`figure_to_dict` for the
        output of :meth:`build_all_figures`.

        Parameters
        ----------
        figures : dict[str, go.Figure]
            Named figures (output of ``build_all_figures``).

        Returns
        -------
        dict[str, dict[str, Any]]
            Same keys, with each value replaced by its dict form.
        """
        return {name: PlotBuilder.figure_to_dict(fig) for name, fig in figures.items()}

    # ──────────────────────────────────────────────────────────────────────
    # Individual plot builders
    # ──────────────────────────────────────────────────────────────────────

    def plot_equity_curve(self) -> list[str]:
        """Plot portfolio value over time.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_equity_curve()
        if fig is None:
            return []
        return self._save_figure(fig, "equity_curve")

    def _build_equity_curve(self) -> go.Figure | None:
        """Build the equity curve figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if data is insufficient.
        """
        if "total_value" not in self._df.columns:
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self._df.index,
                y=self._df["total_value"],
                mode="lines",
                name="Portfolio Value",
                line=dict(color=_COLOURS["primary"], width=2),
            )
        )

        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Period ending (close valuation)"
            if "date_basis" in self._df
            and self._df["date_basis"].eq("period_end_label").all()
            else "Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_white",
            hovermode="x unified",
        )
        _add_grid(fig)
        return fig

    def plot_drawdown(self) -> list[str]:
        """Plot drawdown series with filled area.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_drawdown()
        if fig is None:
            return []
        return self._save_figure(fig, "drawdown")

    def _build_drawdown(self) -> go.Figure | None:
        """Build the drawdown figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if drawdown data is unavailable.
        """
        dd_series: pd.Series | None = self._metrics.get("drawdown_series")
        if dd_series is None or dd_series.empty:
            logger.warning("No drawdown series available; skipping drawdown plot")
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dd_series.index,
                y=_to_pct(dd_series),
                fill="tozeroy",
                mode="lines",
                name="Drawdown",
                line=dict(color=_COLOURS["negative"], width=1.5),
                fillcolor=_COLOURS["drawdown_fill"],
            )
        )

        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            hovermode="x unified",
        )
        _add_grid(fig)
        return fig

    def plot_rolling_sharpe(self) -> list[str]:
        """Plot rolling annualised Sharpe ratio.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_rolling_sharpe()
        if fig is None:
            return []
        return self._save_figure(fig, "rolling_sharpe")

    def _build_rolling_sharpe(self) -> go.Figure | None:
        """Build the rolling Sharpe ratio figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if data is unavailable.
        """
        rolling: pd.Series | None = self._metrics.get("rolling_sharpe")
        if rolling is None or (isinstance(rolling, pd.Series) and rolling.empty):
            logger.warning("No rolling Sharpe data; skipping")
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=rolling.index,
                y=rolling.values,
                mode="lines",
                name="Rolling Sharpe",
                line=dict(color=_COLOURS["secondary"], width=2),
            )
        )
        fig.add_hline(y=0, line_dash="dash", line_color=_COLOURS["neutral"])

        fig.update_layout(
            title="Rolling Sharpe Ratio (annualised)",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            hovermode="x unified",
        )
        _add_grid(fig)
        return fig

    def plot_return_distribution(self) -> list[str]:
        """Plot histogram of period returns with summary statistics.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_return_distribution()
        if fig is None:
            return []
        return self._save_figure(fig, "return_distribution")

    def _build_return_distribution(self) -> go.Figure | None:
        """Build the return distribution histogram figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if return data is empty.
        """
        returns = self._df["return"].dropna()
        if returns.empty:
            return None

        fig = go.Figure()
        fig.add_trace(
            go.Histogram(
                x=_to_pct(returns),
                nbinsx=40,
                marker_color=_COLOURS["primary"],
                opacity=0.75,
                name="Returns",
            )
        )

        mean_ret = float(returns.mean()) * 100
        fig.add_vline(
            x=mean_ret,
            line_dash="dash",
            line_color=_COLOURS["secondary"],
            annotation_text=f"Mean: {mean_ret:.2f}%",
        )

        fig.update_layout(
            title="Period Return Distribution",
            xaxis_title="Return (%)",
            yaxis_title="Frequency",
            template="plotly_white",
        )

        return fig

    def plot_monthly_heatmap(self) -> list[str]:
        """Plot monthly return heatmap (year x month matrix).

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_monthly_heatmap()
        if fig is None:
            return []
        return self._save_figure(fig, "monthly_heatmap")

    def _build_monthly_heatmap(self) -> go.Figure | None:
        """Build the monthly returns heatmap figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if monthly return data is unavailable.
        """
        monthly_df: pd.DataFrame | None = self._metrics.get("monthly_returns")
        if monthly_df is None or (
            isinstance(monthly_df, pd.DataFrame) and monthly_df.empty
        ):
            logger.warning("No monthly returns data; skipping heatmap")
            return None

        z = monthly_df.values * 100
        x_labels = list(monthly_df.columns)
        y_labels = [str(y) for y in monthly_df.index]

        text = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row] for row in z]

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=x_labels,
                y=y_labels,
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="Return (%)"),
            )
        )

        fig.update_layout(
            title="Monthly Returns Heatmap",
            xaxis_title="Month",
            yaxis_title="Year",
            template="plotly_white",
        )

        return fig

    def plot_cumulative_return(self) -> list[str]:
        """Plot cumulative return curve.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_cumulative_return()
        if fig is None:
            return []
        return self._save_figure(fig, "cumulative_return")

    def _build_cumulative_return(self) -> go.Figure | None:
        """Build the cumulative return figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if return data is unavailable.
        """
        if "return" not in self._df.columns:
            return None

        cumret = (1 + self._df["return"]).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=cumret.index,
                y=_to_pct(cumret),
                mode="lines",
                name="Cumulative Return",
                line=dict(color=_COLOURS["positive"], width=2),
            )
        )
        fig.add_hline(y=0, line_dash="dot", line_color=_COLOURS["neutral"])

        fig.update_layout(
            title="Cumulative Return Over Time",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            template="plotly_white",
            hovermode="x unified",
        )
        _add_grid(fig)

        return fig

    def plot_weight_evolution(self) -> list[str]:
        """Plot stacked area of portfolio weight evolution.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_weight_evolution()
        if fig is None:
            return []
        return self._save_figure(fig, "weight_evolution")

    def _build_weight_evolution(self) -> go.Figure | None:
        """Build the weight evolution stacked area figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if weight history data is unavailable.
        """
        wh: pd.DataFrame | None = self._metrics.get("weight_history")
        if wh is None or (isinstance(wh, pd.DataFrame) and wh.empty):
            logger.warning("No weight history data; skipping weight evolution plot")
            return None

        fig = go.Figure()
        for col in wh.columns:
            fig.add_trace(
                go.Scatter(
                    x=wh.index,
                    y=_to_pct(wh[col]),
                    mode="lines",
                    stackgroup="one",
                    name=str(col),
                )
            )

        fig.update_layout(
            title="Portfolio Weight Evolution",
            xaxis_title="Date",
            yaxis_title="Weight (%)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.3,
            ),
        )

        return fig

    def plot_turnover(self) -> list[str]:
        """Plot per-period portfolio turnover as a bar chart.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_turnover()
        if fig is None:
            return []
        return self._save_figure(fig, "turnover")

    def _build_turnover(self) -> go.Figure | None:
        """Build the turnover bar chart figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if turnover data is unavailable.
        """
        turnover: list[float] | None = self._metrics.get("turnover_per_period")
        if not turnover:
            logger.warning("No turnover data; skipping turnover plot")
            return None

        dates = self._df.index[: len(turnover)]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=dates,
                y=[t * 100 for t in turnover],
                marker_color=_COLOURS["primary"],
                name="Turnover",
                opacity=0.75,
            )
        )

        avg = float(np.mean(turnover)) * 100
        fig.add_hline(
            y=avg,
            line_dash="dash",
            line_color=_COLOURS["secondary"],
            annotation_text=f"Avg: {avg:.1f}%",
        )

        fig.update_layout(
            title="Portfolio Turnover per Period",
            xaxis_title="Date",
            yaxis_title="Turnover (%)",
            template="plotly_white",
        )

        return fig

    def plot_underwater(self) -> list[str]:
        """Plot underwater chart (time spent below high-water mark).

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = self._build_underwater()
        if fig is None:
            return []
        return self._save_figure(fig, "underwater")

    def _build_underwater(self) -> go.Figure | None:
        """Build the underwater chart figure.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if ``total_value`` data is unavailable.
        """
        if "total_value" not in self._df.columns:
            return None

        total_values = self._df["total_value"]
        high_water = total_values.cummax()
        underwater = (total_values - high_water) / high_water * 100

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.65, 0.35],
            subplot_titles=("Portfolio Value", "Underwater (%)"),
        )

        fig.add_trace(
            go.Scatter(
                x=total_values.index,
                y=total_values.values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color=_COLOURS["primary"], width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=high_water.index,
                y=high_water.values,
                mode="lines",
                name="High-Water Mark",
                line=dict(color=_COLOURS["neutral"], width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=underwater.index,
                y=underwater.values,
                fill="tozeroy",
                mode="lines",
                name="Underwater",
                line=dict(color=_COLOURS["negative"], width=1),
                fillcolor=_COLOURS["drawdown_fill"],
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            title="Underwater Chart",
            template="plotly_white",
            hovermode="x unified",
            height=650,
        )

        return fig

    # ──────────────────────────────────────────────────────────────────────
    # Live mode chart
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def build_weight_allocation_figure(
        weights: pd.Series,
        title: str = "Recommended Portfolio Allocation",
    ) -> go.Figure | None:
        """Build a pie/bar chart figure for a single weight allocation snapshot.

        This is the headless companion to :meth:`plot_weight_allocation` —
        it builds and returns the Plotly figure without writing anything to
        disk, making it suitable for the web API.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by asset symbol.
        title : str, optional
            Chart title.

        Returns
        -------
        go.Figure | None
            Plotly figure, or ``None`` if all weights are effectively zero.
        """
        non_zero = weights[weights.abs() > 1e-6].sort_values(ascending=False)
        if non_zero.empty:
            return None

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Allocation (%)", "Weight by Asset"),
        )

        fig.add_trace(
            go.Pie(
                labels=non_zero.index.tolist(),
                values=_to_pct(non_zero).tolist(),
                hole=0.35,
                textinfo="label+percent",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=non_zero.index.tolist(),
                y=_to_pct(non_zero).tolist(),
                marker_color=_COLOURS["primary"],
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            title_text=title,
            template="plotly_white",
            showlegend=False,
            height=450,
        )

        return fig

    @staticmethod
    def plot_weight_allocation(
        weights: pd.Series,
        output_dir: Path,
        title: str = "Recommended Portfolio Allocation",
    ) -> list[str]:
        """Plot pie/bar chart for a single weight allocation snapshot.

        Useful for live investment mode output.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by asset symbol.
        output_dir : Path
            Directory to save the plot.
        title : str, optional
            Chart title.

        Returns
        -------
        list[str]
            Saved file paths.
        """
        fig = PlotBuilder.build_weight_allocation_figure(weights, title)
        if fig is None:
            return []

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        html_path = plots_dir / "weight_allocation.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        saved.append(str(html_path))

        try:
            png_path = plots_dir / "weight_allocation.png"
            fig.write_image(str(png_path), width=1200, height=450, scale=2)
            saved.append(str(png_path))
        except Exception:
            logger.debug("Kaleido not available; skipping PNG for weight allocation")

        return saved

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _save_figure(self, fig: go.Figure, name: str) -> list[str]:
        """Save a Plotly figure to HTML and/or PNG.

        Parameters
        ----------
        fig : go.Figure
            Plotly figure to save.
        name : str
            Base filename (without extension).

        Returns
        -------
        list[str]
            Paths of saved files.
        """
        saved: list[str] = []

        if self._save_html:
            html_path = self._plots_dir / f"{name}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            saved.append(str(html_path))
            logger.debug(f"Saved interactive plot: {html_path}")

        if self._save_png:
            try:
                png_path = self._plots_dir / f"{name}.png"
                fig.write_image(str(png_path), width=1200, height=600, scale=2)
                saved.append(str(png_path))
                logger.debug(f"Saved static plot: {png_path}")
            except Exception:
                logger.debug(
                    f"Kaleido not available; skipping PNG for {name}. "
                    "Install kaleido for static image export."
                )

        return saved


def _add_grid(fig: go.Figure) -> None:
    """Add subtle gridlines to a Plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to update.
    """
    fig.update_xaxes(showgrid=True, gridcolor=_COLOURS["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=_COLOURS["grid"])


def _to_pct(series: pd.Series) -> np.ndarray:
    """Convert Series values to percentage as a numpy array.

    Parameters
    ----------
    series : pd.Series
        Input series with decimal values.

    Returns
    -------
    np.ndarray
        Values multiplied by 100.
    """
    return np.asarray(series.values, dtype=float) * 100
