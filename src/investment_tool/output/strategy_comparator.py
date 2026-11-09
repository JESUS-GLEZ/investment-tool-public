"""Strategy comparison — side-by-side analysis of multiple backtests.

Given the raw results (date, total_value, return …) from *N* strategy
backtests run on the **same** universe and time window, this module
computes comparison metrics, generates overlay plots, and produces a
self-contained HTML comparison report.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode

import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.output.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


class StrategyComparator:
    """Compare multiple strategy backtests side-by-side.

    Parameters
    ----------
    results : dict[str, list[dict[str, Any]]]
        ``{strategy_name: backtester_results_list}``.  Each list contains
        per-period dicts with at least ``date``, ``total_value``, ``return``.
    frequency : Frequency
        Data frequency for annualisation.
    output_dir : Path
        Directory for comparison outputs.
    risk_free_rate : float, optional
        Annual risk-free rate, by default 0.0.
    benchmark_name : str | None, optional
        Name of the strategy to treat as benchmark (for excess-return and
        information-ratio calculations).  Defaults to the *first* key in
        *results*.
    """

    def __init__(
        self,
        results: dict[str, list[dict[str, Any]]],
        frequency: Frequency,
        output_dir: Path,
        risk_free_rate: float = 0.0,
        benchmark_name: str | None = None,
        output_mode: OutputMode = OutputMode.DIAGNOSTIC,
    ) -> None:
        self._output_mode = output_mode
        self._raw = results
        self._frequency = frequency
        self._output_dir = output_dir
        self._risk_free_rate = risk_free_rate
        self._ppy = MetricsCalculator._PERIODS_PER_YEAR[frequency]

        # Build per-strategy DataFrames (date-indexed)
        self._frames: dict[str, pd.DataFrame] = {}
        for name, res in results.items():
            df = pd.DataFrame(res)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            if "return" in df.columns:
                df["cumulative_return"] = (1 + df["return"]).cumprod() - 1
            self._frames[name] = df

        self._calculators = {
            name: MetricsCalculator(rows, [], frequency, risk_free_rate)
            for name, rows in results.items()
        }
        self._names = list(self._frames.keys())
        self._benchmark = benchmark_name or (self._names[0] if self._names else "")
        self._date_label = (
            "Period ending (close valuation)"
            if self._frames
            and all(
                "date_basis" in frame
                and frame["date_basis"].eq("period_end_label").all()
                for frame in self._frames.values()
            )
            else "Date"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full comparison pipeline.

        Returns
        -------
        dict[str, Any]
            ``metrics_df``, ``saved_files``, ``output_dir``.
        """
        logger.info(
            "Comparing %d strategies: %s  (benchmark=%s)",
            len(self._names),
            self._names,
            self._benchmark,
        )
        self._output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        metrics_df = self._compute_comparison_metrics()

        # CSV export
        csv_path = self._output_dir / "comparison_metrics.csv"
        metrics_df.to_csv(csv_path)
        saved.append(str(csv_path))

        if self._output_mode.save_chart_exports:
            # Plots
            try:
                saved.extend(self._plot_equity_curves())
            except Exception:
                logger.exception("Failed to generate equity-curve plot")

            try:
                saved.extend(self._plot_rolling_sharpe())
            except Exception:
                logger.exception("Failed to generate rolling-Sharpe plot")

            try:
                saved.extend(self._plot_drawdown())
            except Exception:
                logger.exception("Failed to generate drawdown plot")

            try:
                saved.extend(self._plot_excess_return())
            except Exception:
                logger.exception("Failed to generate excess-return plot")

            # HTML report
            try:
                saved.append(self._build_comparison_report(metrics_df))
            except Exception:
                logger.exception("Failed to generate comparison report")

        logger.info("Strategy comparison complete: %d files saved", len(saved))
        return {
            "metrics_df": metrics_df,
            "metric_definitions": {
                name: calc.compute_performance()["metric_definitions"]
                for name, calc in self._calculators.items()
            },
            "saved_files": saved,
            "output_dir": self._output_dir,
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_comparison_metrics(self) -> pd.DataFrame:
        """Compute per-strategy metrics in a single DataFrame.

        Returns
        -------
        pd.DataFrame
            Rows = metric names, columns = strategy names.
        """
        records: dict[str, dict[str, float]] = {}
        bench_rets = self._get_returns(self._benchmark)

        for name in self._names:
            rets = self._get_returns(name)
            records[name] = self._metrics_for_series(rets, bench_rets, name)

        df = pd.DataFrame(records)
        df.index.name = "metric"
        return df

    def _metrics_for_series(
        self,
        rets: pd.Series,
        bench_rets: pd.Series,
        name: str,
    ) -> dict[str, float]:
        """Compute all comparison metrics for one strategy.

        Parameters
        ----------
        rets : pd.Series
            Period returns for this strategy.
        bench_rets : pd.Series
            Period returns for the benchmark.
        name : str
            Strategy name (for logging).

        Returns
        -------
        dict[str, float]
            Metric name → value.
        """
        performance = self._calculators[name].compute_performance()
        win_rate = float((rets > 0).sum() / len(rets)) if len(rets) else 0.0

        # ── Benchmark-relative metrics ──
        aligned_bench, aligned_strat = bench_rets.align(rets, join="inner")
        excess = aligned_strat - aligned_bench

        # Win rate vs benchmark
        win_vs_bench = float((excess > 0).sum() / len(excess)) if len(excess) else 0.0

        # Tracking error
        te = float(excess.std()) * np.sqrt(self._ppy) if len(excess) > 1 else 0.0

        # Information ratio
        info_ratio = float(excess.mean()) * self._ppy / te if te > 0 else 0.0

        return {
            **{
                key: float(performance[key])
                for key in (
                    "total_return",
                    "annualized_return",
                    "annualized_volatility",
                    "sharpe_ratio",
                    "sortino_ratio",
                    "max_drawdown",
                    "calmar_ratio",
                )
            },
            "win_rate": win_rate,
            "win_rate_vs_benchmark": win_vs_bench,
            "tracking_error": te,
            "information_ratio": info_ratio,
        }

    # ------------------------------------------------------------------
    # Plots (Plotly)
    # ------------------------------------------------------------------

    def _plot_equity_curves(self) -> list[str]:
        """Overlay cumulative-return curves for all strategies."""
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        fig = go.Figure()
        for name, df in self._frames.items():
            if "cumulative_return" in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df["cumulative_return"],
                        mode="lines",
                        name=name,
                    )
                )
        fig.update_layout(
            title="Cumulative Return — Strategy Comparison",
            xaxis_title=self._date_label,
            yaxis_title="Cumulative Return",
            yaxis_tickformat=".1%",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return self._save_plotly(fig, "equity_curves")

    def _plot_rolling_sharpe(self, window: int | None = None) -> list[str]:
        """Plot rolling Sharpe ratio for each strategy."""
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        if window is None:
            window = max(6, self._ppy // 4)

        fig = go.Figure()
        for name in self._names:
            rets = self._get_returns(name)
            if len(rets) < window:
                continue
            roll_sharpe = self._calculators[name].rolling_sharpe(window)
            fig.add_trace(
                go.Scatter(
                    x=roll_sharpe.index,
                    y=roll_sharpe,
                    mode="lines",
                    name=name,
                )
            )
        fig.update_layout(
            title=f"Rolling Sharpe Ratio ({window}-period window)",
            xaxis_title=self._date_label,
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return self._save_plotly(fig, "rolling_sharpe")

    def _plot_drawdown(self) -> list[str]:
        """Plot underwater (drawdown) chart for each strategy."""
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        fig = go.Figure()
        for name in self._names:
            dd = self._calculators[name].compute_performance()["drawdown_series"]
            fig.add_trace(
                go.Scatter(
                    x=dd.index,
                    y=dd,
                    mode="lines",
                    name=name,
                    fill="tozeroy",
                    opacity=0.4,
                )
            )
        fig.update_layout(
            title="Drawdown — Strategy Comparison",
            xaxis_title=self._date_label,
            yaxis_title="Drawdown",
            yaxis_tickformat=".1%",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return self._save_plotly(fig, "drawdown")

    def _plot_excess_return(self) -> list[str]:
        """Plot cumulative excess return vs benchmark."""
        import plotly.graph_objects as go  # type: ignore[import-untyped]

        bench_rets = self._get_returns(self._benchmark)
        fig = go.Figure()
        for name in self._names:
            if name == self._benchmark:
                continue
            rets = self._get_returns(name)
            aligned_b, aligned_s = bench_rets.align(rets, join="inner")
            excess_cum = (1 + aligned_s).cumprod() / (1 + aligned_b).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=excess_cum.index,
                    y=excess_cum,
                    mode="lines",
                    name=f"{name} vs {self._benchmark}",
                )
            )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"Cumulative Excess Return vs {self._benchmark}",
            xaxis_title=self._date_label,
            yaxis_title="Excess Return",
            yaxis_tickformat=".1%",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return self._save_plotly(fig, "excess_return")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _build_comparison_report(self, metrics_df: pd.DataFrame) -> str:
        """Generate a self-contained HTML comparison report.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Comparison metrics (rows=metric, cols=strategy).

        Returns
        -------
        str
            Path to the saved HTML file.
        """
        sections: list[str] = []

        # Metrics table
        sections.append(self._html_metrics_table(metrics_df))

        # Embed PNGs from plots/comparison/
        plots_dir = self._output_dir / "comparison_plots"
        if plots_dir.exists():
            images: list[str] = []
            for png in sorted(plots_dir.glob("*.png")):
                try:
                    data = base64.b64encode(png.read_bytes()).decode("ascii")
                    title = png.stem.replace("_", " ").title()
                    images.append(
                        f'<div class="plot"><h3>{title}</h3>'
                        f'<img src="data:image/png;base64,{data}" '
                        f'alt="{png.stem}"></div>'
                    )
                except Exception:
                    logger.debug("Could not embed plot: %s", png)
            if images:
                sections.append(
                    f'<div class="card"><h2>Charts</h2>' f'{"".join(images)}</div>'
                )

        body = "\n".join(sections)
        html = _COMPARISON_HTML_TEMPLATE.format(
            title="Strategy Comparison Report", body=body
        )
        path = self._output_dir / "comparison_report.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved comparison report: %s", path)
        return str(path)

    def _html_metrics_table(self, metrics_df: pd.DataFrame) -> str:
        """Render the metrics DataFrame as an HTML card.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Comparison metrics.

        Returns
        -------
        str
            HTML fragment.
        """
        header = (
            "<tr><th>Metric</th>"
            + "".join(f"<th>{c}</th>" for c in metrics_df.columns)
            + "</tr>"
        )
        rows: list[str] = []
        _PCT_METRICS = {
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "win_rate",
            "win_rate_vs_benchmark",
            "tracking_error",
        }
        for metric, vals in metrics_df.iterrows():
            label = str(metric)
            if metric == "annualized_return":
                bases = [
                    calc.compute_performance()["metric_definitions"][
                        "annualized_return"
                    ]
                    for calc in self._calculators.values()
                ]
                label = (
                    "CAGR"
                    if bases and all(b == "cagr" for b in bases)
                    else "Annualized compound return"
                )
            cells = f"<td><strong>{label}</strong></td>"
            for v in vals:
                if pd.isna(v):
                    cells += "<td>—</td>"
                elif str(metric) in _PCT_METRICS:
                    cells += f"<td>{float(v) * 100:.2f}%</td>"
                else:
                    cells += f"<td>{float(v):.4f}</td>"
            rows.append(f"<tr>{cells}</tr>")

        table = f"<table>{header}{''.join(rows)}</table>"
        return f'<div class="card"><h2>Performance Metrics</h2>{table}</div>'

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_returns(self, name: str) -> pd.Series:
        """Get the period-return Series for a strategy.

        Parameters
        ----------
        name : str
            Strategy name.

        Returns
        -------
        pd.Series
            Period returns.
        """
        df = self._frames.get(name)
        if df is None or "return" not in df.columns:
            return pd.Series(dtype=float)
        return df["return"].dropna()

    def _save_plotly(self, fig: Any, name: str) -> list[str]:
        """Save a Plotly figure as HTML and optionally PNG.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            Figure to save.
        name : str
            Base filename (without extension).

        Returns
        -------
        list[str]
            Saved file paths.
        """
        saved: list[str] = []
        if not self._output_mode.save_chart_exports:
            return saved
        plots_dir = self._output_dir / "comparison_plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        html_path = plots_dir / f"{name}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn")
        saved.append(str(html_path))

        try:
            png_path = plots_dir / f"{name}.png"
            fig.write_image(str(png_path), width=1200, height=600, scale=2)
            saved.append(str(png_path))
        except Exception:
            logger.debug("PNG export unavailable for %s (kaleido not installed?)", name)

        return saved


# ── HTML template ───────────────────────────────────────────────────────────

_COMPARISON_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  :root {{ --primary: #1f77b4; --bg: #f8f9fa; --card: #ffffff; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
          sans-serif; background: var(--bg); color: #333; padding: 2rem; }}
  h1 {{ text-align: center; color: var(--primary); margin-bottom: 2rem; }}
  .card {{ background: var(--card); border-radius: 8px; padding: 1.5rem;
           margin-bottom: 1.5rem; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .card h2 {{ color: var(--primary); border-bottom: 2px solid var(--primary);
              padding-bottom: .4rem; margin-bottom: 1rem; font-size: 1.2rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
  td, th {{ padding: .4rem .8rem; text-align: left; border-bottom: 1px solid #eee; }}
  th {{ background: #f0f4f8; }}
  tr:hover {{ background: #f5f7fa; }}
  .plot img {{ max-width: 100%; height: auto; margin-top: .5rem; }}
  .plot h3 {{ font-size: 1rem; color: #555; }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""
