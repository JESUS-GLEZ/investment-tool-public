"""
Top-level output façade for the Investment Tool.

This module provides the ``OutputManager`` class which orchestrates all output
generation — metrics computation, interactive/static plots, data exports,
and HTML reports — for both backtesting and live investment modes.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from investment_tool.constants import Frequency
from investment_tool.output.export import DataExporter
from investment_tool.output.metrics import MetricsCalculator
from investment_tool.output.plots import PlotBuilder
from investment_tool.output.report import ReportBuilder

logger = logging.getLogger(__name__)


class OutputManager:
    """Façade that orchestrates all output generation.

    Provides a single entry-point for both backtesting and live investment
    modes. Internally delegates to ``MetricsCalculator``, ``PlotBuilder``,
    ``DataExporter``, and ``ReportBuilder``.

    Parameters
    ----------
    output_dir : Path
        Root directory for all output files.
    frequency : Frequency
        Data frequency for annualisation calculations.
    risk_free_rate : float, optional
        Annual risk-free rate, by default 0.0.
    save_html_plots : bool, optional
        Whether to save interactive HTML charts, by default True.
    save_png_plots : bool, optional
        Whether to save static PNG charts, by default True.
    generate_report : bool, optional
        Whether to generate the HTML summary report, by default True.
    """

    def __init__(
        self,
        output_dir: Path,
        frequency: Frequency,
        risk_free_rate: float = 0.0,
        save_html_plots: bool = True,
        save_png_plots: bool = True,
        generate_report: bool = True,
        output_mode: OutputMode = OutputMode.DIAGNOSTIC,
    ) -> None:
        self._output_dir = output_dir
        self._frequency = frequency
        self._risk_free_rate = risk_free_rate
        self._save_html = save_html_plots and output_mode.save_chart_exports
        self._save_png = save_png_plots and output_mode.save_chart_exports
        self._generate_report = generate_report and output_mode.save_chart_exports
        self._output_mode = output_mode

        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Backtesting output
    # ------------------------------------------------------------------

    def generate_backtest_output(
        self,
        results: list[dict[str, Any]],
        portfolio_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Generate full output suite for a completed backtest.

        Computes comprehensive metrics, generates plots, exports data
        files, and builds an HTML report.

        Parameters
        ----------
        results : list[dict[str, Any]]
            Per-period snapshots from the backtester (date, total_value,
            return).
        portfolio_history : list[dict[str, Any]]
            Portfolio rebalancing history from ``Portfolio.history``.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - ``metrics`` — full metrics dictionary
            - ``results_df`` — date-indexed results DataFrame
            - ``saved_files`` — list of all saved file paths
            - ``output_dir`` — output directory path
        """
        logger.info(f"Generating backtest output in {self._output_dir}")
        saved_files: list[str] = []

        # 1. Compute metrics
        calc = MetricsCalculator(
            results=results,
            portfolio_history=portfolio_history,
            frequency=self._frequency,
            risk_free_rate=self._risk_free_rate,
        )
        metrics = calc.compute_all()

        # 2. Build results DataFrame for plots and exports
        results_df = self._build_results_df(results)

        # 3. Build rebalancing detail (selected assets + prices per period)
        rebalancing_detail = self._build_rebalancing_detail(results)

        # 4. Log key metrics
        self._log_metrics_summary(metrics)

        # 5. Plots
        try:
            plot_builder = PlotBuilder(
                results_df=results_df,
                metrics=metrics,
                output_dir=self._output_dir,
                save_html=self._save_html,
                save_png=self._save_png,
            )
            if self._save_html or self._save_png:
                saved_files.extend(plot_builder.build_all())
        except Exception:
            logger.exception("Failed to generate plots")

        # 6. Data exports
        try:
            exporter = DataExporter(self._output_dir)
            saved_files.extend(
                exporter.export_all(results_df, metrics, rebalancing_detail)
            )
        except Exception:
            logger.exception("Failed to export data files")

        # 7. HTML report
        if self._generate_report:
            try:
                report = ReportBuilder(
                    metrics=metrics,
                    output_dir=self._output_dir,
                    rebalancing_detail=rebalancing_detail,
                )
                saved_files.append(report.build())
            except Exception:
                logger.exception("Failed to generate HTML report")

        logger.info(f"Backtest output complete: {len(saved_files)} files saved")

        return {
            "metrics": metrics,
            "results_df": results_df,
            "saved_files": saved_files,
            "output_dir": self._output_dir,
        }

    # ------------------------------------------------------------------
    # Live investment output
    # ------------------------------------------------------------------

    def generate_live_output(
        self,
        optimal_weights: pd.Series,
        portfolio_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate output for live investment mode.

        Exports optimal weights, portfolio snapshot, concentration analysis,
        and an allocation chart.

        Parameters
        ----------
        optimal_weights : pd.Series
            Recommended portfolio weights indexed by asset symbol.
        portfolio_snapshot : dict[str, Any] | None, optional
            Current portfolio state for export.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - ``concentration`` — weight concentration metrics
            - ``optimal_weights`` — the weights Series
            - ``saved_files`` — list of saved file paths
            - ``output_dir`` — output directory path
        """
        logger.info(f"Generating live output in {self._output_dir}")
        saved_files: list[str] = []

        # 1. Concentration analysis
        concentration = MetricsCalculator.compute_weight_concentration(optimal_weights)
        logger.info(
            f"Live allocation: {concentration['num_active_positions']} assets, "
            f"HHI={concentration['hhi']:.4f}, "
            f"Top-3={concentration['top3_weight']:.2%}"
        )

        # 2. Export weights & snapshot
        try:
            exporter = DataExporter(self._output_dir)
            saved_files.extend(exporter.export_live_weights(optimal_weights))
            if portfolio_snapshot:
                saved_files.extend(
                    exporter.export_portfolio_snapshot(portfolio_snapshot)
                )
        except Exception:
            logger.exception("Failed to export live data")

        if self._output_mode.save_chart_exports:
            # 3. Allocation plot
            try:
                saved_files.extend(
                    PlotBuilder.plot_weight_allocation(
                        weights=optimal_weights,
                        output_dir=self._output_dir,
                    )
                )
            except Exception:
                logger.exception("Failed to generate allocation plot")

        logger.info(f"Live output complete: {len(saved_files)} files saved")

        return {
            "concentration": concentration,
            "optimal_weights": optimal_weights,
            "saved_files": saved_files,
            "output_dir": self._output_dir,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_results_df(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a date-indexed results DataFrame with cumulative return.

        Parameters
        ----------
        results : list[dict[str, Any]]
            Raw per-period results.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame.
        """
        df = pd.DataFrame(results)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        if "return" in df.columns:
            df["cumulative_return"] = (1 + df["return"]).cumprod() - 1

        # Build a human-readable summary of selected assets per period
        if "selected_assets" in df.columns:
            df["selected_assets_summary"] = df["selected_assets"].apply(
                lambda d: (
                    ", ".join(f"{s}:{w:.2%}" for s, w in d.items())
                    if isinstance(d, dict) and d
                    else ""
                )
            )

        # Drop nested dicts that don't belong in a flat CSV/DataFrame
        for col in ("selected_assets", "asset_prices"):
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    @staticmethod
    def _build_rebalancing_detail(
        results: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Build a flat DataFrame with per-asset weights and prices per period.

        For each rebalancing period and each selected asset, one row is
        produced containing the asset's weight, its price at that period,
        and its price at the next period (to verify the price evolution).

        Parameters
        ----------
        results : list[dict[str, Any]]
            Per-period snapshots from the backtester containing
            ``selected_assets`` and ``asset_prices``.

        Returns
        -------
        pd.DataFrame
            Columns: ``date``, ``symbol``, ``weight``, ``price``,
            ``next_price``, ``price_change_pct``.
        """
        rows: list[dict[str, Any]] = []

        for i, snap in enumerate(results):
            dt = snap.get("date")
            selected = snap.get("selected_assets", {})
            prices = snap.get("asset_prices", {})

            # Look-ahead: next period's prices (for verification only)
            next_prices: dict[str, float] = {}
            if i + 1 < len(results):
                next_prices = results[i + 1].get("asset_prices", {})

            for symbol, weight in selected.items():
                price = prices.get(symbol)
                next_price = next_prices.get(symbol)
                change_pct = (
                    ((next_price - price) / price) * 100.0
                    if price and next_price and price != 0
                    else None
                )
                rows.append(
                    {
                        "date": dt,
                        "period_start": snap.get("period_start"),
                        "period_end": snap.get("period_end"),
                        "execution_time": snap.get("execution_time"),
                        "decision_time": snap.get("decision_time"),
                        "symbol": symbol,
                        "weight": weight,
                        "price": price,
                        "next_price": next_price,
                        "price_change_pct": change_pct,
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "date",
                    "symbol",
                    "weight",
                    "price",
                    "next_price",
                    "price_change_pct",
                ]
            )

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def _log_metrics_summary(metrics: dict[str, Any]) -> None:
        """Log a concise summary of key metrics.

        Parameters
        ----------
        metrics : dict[str, Any]
            Computed metrics dictionary.
        """
        logger.info("----- Backtest Performance Metrics -----")
        logger.info(f"Total Return:          {metrics.get('total_return', 0):.4f}")
        logger.info(f"Annualised Return:     {metrics.get('annualized_return', 0):.4f}")
        logger.info(
            f"Annualised Volatility: {metrics.get('annualized_volatility', 0):.4f}"
        )
        logger.info(f"Sharpe Ratio:          {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Sortino Ratio:         {metrics.get('sortino_ratio', 0):.4f}")
        logger.info(f"Max Drawdown:          {metrics.get('max_drawdown', 0):.4f}")
        logger.info(f"Calmar Ratio:          {metrics.get('calmar_ratio', 0):.4f}")
        logger.info(f"Win Rate:              {metrics.get('win_rate', 0):.4f}")
        logger.info(f"Profit Factor:         {metrics.get('profit_factor', 0):.4f}")
        logger.info(f"Avg Turnover:          {metrics.get('average_turnover', 0):.4f}")
        logger.info(f"Final HHI:             {metrics.get('final_hhi', 0):.4f}")
        logger.info("----------------------------------------")
