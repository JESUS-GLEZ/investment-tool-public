"""
Performance metrics computation for investment analysis.

This module provides the MetricsCalculator class which computes a comprehensive
set of performance metrics from backtesting results and portfolio history,
including risk-adjusted returns, drawdown analysis, turnover, and
concentration measures.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive performance metrics calculator.

    Computes a full suite of investment performance metrics from backtest
    results and portfolio trade history.

    Parameters
    ----------
    results : list[dict[str, Any]]
        List of per-period snapshots, each containing 'date', 'total_value',
        and 'return' keys.
    portfolio_history : list[dict[str, Any]]
        Portfolio rebalancing history from ``Portfolio.history``, each entry
        containing 'date', 'cash', 'positions', 'trades', 'prices',
        'pre_rebalance_value', 'post_rebalance_value'.
    frequency : Frequency
        Data frequency (DAILY, WEEKLY, MONTHLY).
    risk_free_rate : float, optional
        Annual risk-free rate for ratio calculations, by default 0.0.
    """

    _PERIODS_PER_YEAR: dict[Frequency, int] = {
        Frequency.DAILY: 252,
        Frequency.WEEKLY: 52,
        Frequency.MONTHLY: 12,
    }

    def __init__(
        self,
        results: list[dict[str, Any]],
        portfolio_history: list[dict[str, Any]],
        frequency: Frequency,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.frequency = frequency
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = self._PERIODS_PER_YEAR[frequency]

        self._results_df = self._build_results_df(results)
        self._results_df["return"] = pd.to_numeric(self._results_df["return"]).astype(
            float
        )
        self._results_df["total_value"] = pd.to_numeric(
            self._results_df["total_value"]
        ).astype(float)
        self._portfolio_history = portfolio_history
        self._returns = self._results_df["return"]
        self._cumulative = (1 + self._returns).cumprod()
        self._total_values = self._results_df["total_value"]
        if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
            raise ValueError("Annual risk-free rate must be finite and greater than -1")
        if not np.isfinite(self._returns).all() or (self._returns < -1).any():
            raise ValueError("Period returns must be finite and at least -100%")
        if not np.isfinite(self._total_values).all() or (self._total_values < 0).any():
            raise ValueError("Portfolio values must be finite and nonnegative")
        self._rf_per_period = (1 + risk_free_rate) ** (1 / self.periods_per_year) - 1

    @staticmethod
    def _build_results_df(results: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a date-indexed DataFrame from raw results.

        Parameters
        ----------
        results : list[dict[str, Any]]
            Raw per-period result dictionaries.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by date with total_value and return columns.
        """
        df = (
            pd.DataFrame(results)
            if results
            else pd.DataFrame(columns=["date", "total_value", "return"])
        )
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df

    def compute_all(self) -> dict[str, Any]:
        """Compute the full suite of performance metrics.

        Returns
        -------
        dict[str, Any]
            Dictionary with all computed metrics. Keys include:
            - Basic: average_return, std_return, annualized_return,
              annualized_volatility, sharpe_ratio
            - Drawdown: max_drawdown, max_drawdown_duration_periods,
              max_drawdown_start, max_drawdown_end
            - Risk-adjusted: calmar_ratio, sortino_ratio
            - Win/loss: win_rate, profit_factor, best_period, worst_period
            - Rolling: rolling_sharpe (pd.Series)
            - Breakdown: monthly_returns, yearly_returns
            - Portfolio: turnover_per_period, average_turnover,
              total_turnover, total_transaction_costs, weight_history
            - Concentration: final_hhi, final_top3_weight
            - Summary: total_return, total_periods, initial_value, final_value
        """
        logger.info("Computing comprehensive performance metrics")

        metrics = self.compute_performance()
        # Win/loss
        metrics.update(self._compute_win_loss_metrics())
        # Rolling
        metrics.update(self._compute_rolling_metrics())
        # Breakdown
        metrics.update(self._compute_return_breakdowns())
        # Portfolio / turnover
        metrics.update(self._compute_portfolio_metrics())

        logger.info(
            f"Metrics computed: Sharpe={metrics.get('sharpe_ratio', 0):.4f}, "
            f"MaxDD={metrics.get('max_drawdown', 0):.4f}, "
            f"Sortino={metrics.get('sortino_ratio', 0):.4f}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Basic metrics
    # ------------------------------------------------------------------

    def compute_performance(self) -> dict[str, Any]:
        """Shared realized performance for backtester, reports and comparisons."""
        metrics = self._compute_summary()
        metrics.update(self._compute_basic_metrics(metrics["total_return"]))
        metrics.update(self._compute_drawdown_metrics())
        metrics.update(self._compute_risk_adjusted_ratios(metrics))
        return metrics

    def _annualization_years(self) -> tuple[float, str]:
        """Use explicit complete calendar intervals; legacy rows use period count."""
        if self._results_df.empty:
            return float("nan"), "unavailable"
        frame = self._results_df
        if "date_basis" in frame and frame["date_basis"].eq("period_end_label").all():
            starts = pd.to_datetime(frame["period_start"])
            ends = pd.to_datetime(frame["period_end"]) + pd.Timedelta(days=1)
            if starts.isna().any() or ends.isna().any() or (starts >= ends).any():
                raise ValueError("Invalid result period boundaries for CAGR")
            if not starts.is_monotonic_increasing or starts.duplicated().any():
                raise ValueError("Result periods must be unique and sorted")
            if (
                len(starts) > 1
                and (starts.iloc[1:].to_numpy() < ends.iloc[:-1].to_numpy()).any()
            ):
                raise ValueError("Simulated result periods cannot overlap")
            return float(
                (ends.iloc[-1] - starts.iloc[0]) / pd.Timedelta(days=365.25)
            ), "calendar_act_365_25"
        return len(self._returns) / self.periods_per_year, "period_count"

    def _compute_basic_metrics(self, total_return: float) -> dict[str, Any]:
        """Compound growth and arithmetic excess-return Sharpe are distinct."""
        avg_ret = float(self._returns.mean())
        std_ret = float(self._returns.std(ddof=1))
        years, basis = self._annualization_years()
        growth = 1 + total_return
        ann_ret = growth ** (1 / years) - 1 if years > 0 else float("nan")
        ann_vol = std_ret * np.sqrt(self.periods_per_year)
        excess_mean = avg_ret - self._rf_per_period
        sharpe = (
            excess_mean / std_ret * np.sqrt(self.periods_per_year)
            if std_ret > np.finfo(float).eps
            else float("nan")
        )
        return {
            "average_return": avg_ret,
            "std_return": std_ret,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "metric_definitions": {
                "version": 2,
                "annualized_return": "cagr"
                if basis == "calendar_act_365_25"
                else "geometric_period_return",
                "annualization_basis": basis,
                "elapsed_years": years,
                "periods_per_year": self.periods_per_year,
                "annual_risk_free_rate": self.risk_free_rate,
                "sharpe": "arithmetic_excess_sample_std_sqrt_periods_per_year",
                "sortino": "sample_downside_all_observations_sqrt_frequency_approximation",
            },
        }

    # ------------------------------------------------------------------
    # Drawdown
    # ------------------------------------------------------------------

    def _compute_drawdown_metrics(self) -> dict[str, Any]:
        """Compute drawdown-related metrics.

        Returns
        -------
        dict[str, Any]
            max_drawdown, max_drawdown_duration_periods,
            max_drawdown_start, max_drawdown_end, drawdown_series.
        """
        cumulative_max = self._cumulative.cummax().clip(lower=1.0)
        drawdown = self._cumulative / cumulative_max - 1
        max_dd = float(drawdown.min()) if len(drawdown) else float("nan")
        peak_date: str | None = None
        trough_date: str | None = None
        duration = 0
        if max_dd < 0:
            trough_pos = int(drawdown.to_numpy().argmin())
            peak_values = self._cumulative.iloc[: trough_pos + 1]
            if float(peak_values.max()) < 1:
                peak_pos = -1
                peak_date = str(
                    self._results_df.iloc[0].get("period_start", "initial capital")
                )
            else:
                peak_pos = int(peak_values.to_numpy().argmax())
                peak_date = str(self._returns.index[peak_pos])
            trough_date = str(self._returns.index[trough_pos])
            recovered = (
                self._cumulative.iloc[trough_pos:] >= cumulative_max.iloc[trough_pos]
            )
            recovery_pos = (
                trough_pos + int(recovered.to_numpy().argmax())
                if recovered.any()
                else len(drawdown) - 1
            )
            duration = recovery_pos - peak_pos
        return {
            "max_drawdown": max_dd,
            "max_drawdown_duration_periods": duration,
            "max_drawdown_start": peak_date,
            "max_drawdown_end": trough_date,
            "drawdown_series": drawdown,
        }

    # ------------------------------------------------------------------
    # Risk-adjusted ratios
    # ------------------------------------------------------------------

    def _compute_risk_adjusted_ratios(
        self, base_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Compute Calmar and Sortino ratios.

        Parameters
        ----------
        base_metrics : dict[str, Any]
            Previously computed metrics (needs annualized_return, max_drawdown).

        Returns
        -------
        dict[str, Any]
            calmar_ratio, sortino_ratio.
        """
        ann_ret = base_metrics["annualized_return"]
        max_dd = base_metrics.get("max_drawdown", 0.0)

        calmar = ann_ret / abs(max_dd) if max_dd < 0 else float("nan")
        excess = self._returns - self._rf_per_period
        downside = float(np.sqrt(excess.clip(upper=0).pow(2).mean()))
        sortino = (
            float(excess.mean()) / downside * np.sqrt(self.periods_per_year)
            if downside > np.finfo(float).eps
            else float("nan")
        )

        return {
            "calmar_ratio": calmar,
            "sortino_ratio": sortino,
        }

    # ------------------------------------------------------------------
    # Win/loss
    # ------------------------------------------------------------------

    def _compute_win_loss_metrics(self) -> dict[str, Any]:
        """Compute win rate, profit factor, and extreme periods.

        Returns
        -------
        dict[str, Any]
            win_rate, profit_factor, best_period, worst_period,
            best_period_date, worst_period_date.
        """
        positive = self._returns[self._returns > 0]
        negative = self._returns[self._returns < 0]
        total_periods = len(self._returns)

        win_rate = len(positive) / total_periods if total_periods > 0 else 0.0

        total_gains = float(positive.sum()) if len(positive) > 0 else 0.0
        total_losses = float(abs(negative.sum())) if len(negative) > 0 else 0.0
        profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

        best_ret = float(self._returns.max())
        worst_ret = float(self._returns.min())
        best_date = str(self._returns.idxmax()) if total_periods else None
        worst_date = str(self._returns.idxmin()) if total_periods else None

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "best_period": best_ret,
            "worst_period": worst_ret,
            "best_period_date": best_date,
            "worst_period_date": worst_date,
        }

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def _compute_rolling_metrics(self) -> dict[str, Any]:
        """Compute rolling Sharpe ratio.

        Uses a 6-period rolling window (6 months for monthly data,
        6 weeks for weekly, etc.).

        Returns
        -------
        dict[str, Any]
            rolling_sharpe (pd.Series).
        """
        window = min(6, len(self._returns))
        if window < 2:
            return {"rolling_sharpe": pd.Series(dtype=float)}

        return {"rolling_sharpe": self.rolling_sharpe(window)}

    def rolling_sharpe(self, window: int) -> pd.Series:
        """Annualized arithmetic excess Sharpe with sample volatility."""
        if window < 2:
            return pd.Series(dtype=float)
        excess = self._returns - self._rf_per_period
        mean = excess.rolling(window).mean()
        std = excess.rolling(window).std(ddof=1)
        result: pd.Series = pd.Series(
            (mean / std.where(std > np.finfo(float).eps))
            * np.sqrt(self.periods_per_year)
        )
        result.dropna(inplace=True)
        return result

    # ------------------------------------------------------------------
    # Return breakdowns
    # ------------------------------------------------------------------

    def _compute_return_breakdowns(self) -> dict[str, Any]:
        """Compute monthly and yearly return breakdowns.

        Returns
        -------
        dict[str, Any]
            monthly_returns (pd.DataFrame with year rows, month columns),
            yearly_returns (pd.Series).
        """
        if self._results_df.index.empty:
            return {
                "monthly_returns": pd.DataFrame(),
                "yearly_returns": pd.Series(dtype=float),
            }

        # Ensure datetime index
        idx = pd.DatetimeIndex(self._results_df.index)
        returns = self._returns.copy()
        returns.index = idx

        # Monthly: compound returns within each month
        monthly = (1 + returns).groupby([idx.year, idx.month]).prod() - 1
        monthly.index = pd.MultiIndex.from_tuples(
            monthly.index, names=["year", "month"]
        )

        # Pivot to year x month matrix
        monthly_df = monthly.unstack(level="month")
        monthly_df.columns = [_month_name(int(m)) for m in monthly_df.columns]

        # Yearly
        yearly = (1 + returns).groupby(idx.year).prod() - 1
        yearly.index.name = "year"

        return {
            "monthly_returns": monthly_df,
            "yearly_returns": yearly,
        }

    # ------------------------------------------------------------------
    # Portfolio / turnover metrics
    # ------------------------------------------------------------------

    def _compute_portfolio_metrics(self) -> dict[str, Any]:
        """Compute turnover and weight history from portfolio history.

        Returns
        -------
        dict[str, Any]
            turnover_per_period (list[float]), average_turnover,
            total_turnover, total_transaction_costs, weight_history (pd.DataFrame).
        """
        if not self._portfolio_history:
            return {
                "turnover_per_period": [],
                "average_turnover": 0.0,
                "total_turnover": 0.0,
                "total_transaction_costs": None,
                "weight_history": pd.DataFrame(),
            }

        turnover_list: list[float] = []
        weight_records: list[dict[str, Any]] = []

        for entry in self._portfolio_history:
            # Turnover: sum of absolute trade values / portfolio value
            trades = entry.get("trades", {})
            prices = entry.get("prices", {})
            post_value = entry.get("post_rebalance_value", 1.0)

            if trades and prices and post_value > 0:
                trade_value = sum(
                    abs(float(qty)) * float(prices.get(sym, 0.0))
                    for sym, qty in trades.items()
                )
                turnover_list.append(trade_value / post_value)
            else:
                turnover_list.append(0.0)

            # Weight snapshot
            positions = entry.get("positions", {})
            if positions and prices and post_value > 0:
                weights: dict[str, Any] = {"date": entry.get("date")}
                for sym, qty in positions.items():
                    price = float(prices.get(sym, 0.0))
                    weights[sym] = (float(qty) * price) / post_value
                weight_records.append(weights)

        weight_df = pd.DataFrame(weight_records)
        if "date" in weight_df.columns:
            weight_df["date"] = pd.to_datetime(weight_df["date"])
            weight_df = weight_df.set_index("date")
        weight_df = weight_df.fillna(0.0)

        avg_turnover = float(np.mean(turnover_list)) if turnover_list else 0.0
        total_turnover = float(np.sum(turnover_list))
        costs = [entry.get("transaction_costs") for entry in self._portfolio_history]
        valid_costs = [
            float(cost)
            for cost in costs
            if isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and np.isfinite(cost)
            and cost >= 0
        ]

        return {
            "turnover_per_period": turnover_list,
            "average_turnover": avg_turnover,
            "total_turnover": total_turnover,
            "total_transaction_costs": (
                float(sum(valid_costs))
                if len(valid_costs) == len(self._portfolio_history)
                else None
            ),
            "weight_history": weight_df,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(self) -> dict[str, Any]:
        """Compute summary statistics.

        Returns
        -------
        dict[str, Any]
            total_return, total_periods, initial_value, final_value,
            final_hhi, final_top3_weight.
        """
        initial = float("nan")
        final = (
            float(self._total_values.iloc[-1])
            if len(self._total_values)
            else float("nan")
        )
        if len(self._returns):
            first = self._results_df.iloc[0]
            if pd.notna(first.get("starting_value")):
                initial = float(first["starting_value"])
            elif 1 + float(self._returns.iloc[0]) > 0:
                initial = float(self._total_values.iloc[0]) / (
                    1 + float(self._returns.iloc[0])
                )
        total_ret = (
            float(self._cumulative.iloc[-1] - 1) if len(self._returns) else float("nan")
        )
        if (
            np.isfinite(initial)
            and initial > 0
            and not np.allclose(
                self._total_values.to_numpy(dtype=float),
                initial * self._cumulative.to_numpy(dtype=float),
                rtol=1e-8,
                atol=1e-8,
            )
        ):
            raise ValueError(
                "Portfolio values do not reconcile with reported period returns"
            )
        if np.isfinite(initial) and initial > 0:
            total_ret = final / initial - 1

        # Concentration of final portfolio
        hhi = 0.0
        top3_weight = 0.0
        if self._portfolio_history:
            last = self._portfolio_history[-1]
            positions = last.get("positions", {})
            prices = last.get("prices", {})
            post_val = last.get("post_rebalance_value", 1.0)
            if positions and prices and post_val > 0:
                weights = sorted(
                    [
                        (float(qty) * float(prices.get(sym, 0.0))) / post_val
                        for sym, qty in positions.items()
                    ],
                    reverse=True,
                )
                hhi = float(sum(w**2 for w in weights))
                top3_weight = float(sum(weights[:3]))

        return {
            "total_return": total_ret,
            "total_periods": len(self._returns),
            "initial_value": initial,
            "final_value": final,
            "final_hhi": hhi,
            "final_top3_weight": top3_weight,
        }

    # ------------------------------------------------------------------
    # Live mode helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_weight_concentration(weights: pd.Series) -> dict[str, Any]:
        """Compute concentration metrics for a weight vector.

        Useful for live investment mode analysis of recommended allocation.

        Parameters
        ----------
        weights : pd.Series
            Portfolio weights indexed by asset symbol.

        Returns
        -------
        dict[str, Any]
            hhi, top3_weight, num_active_positions, effective_n,
            weight_distribution (dict of symbol -> weight for non-zero).
        """
        non_zero = weights[weights.abs() > 1e-8]
        sorted_w = non_zero.sort_values(ascending=False)

        hhi = float((sorted_w**2).sum())
        top3 = float(sorted_w.head(3).sum())
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        return {
            "hhi": hhi,
            "top3_weight": top3,
            "num_active_positions": len(non_zero),
            "effective_n": effective_n,
            "weight_distribution": sorted_w.to_dict(),
        }


def _month_name(month_num: int) -> str:
    """Convert month number to abbreviated name.

    Parameters
    ----------
    month_num : int
        Month number (1-12).

    Returns
    -------
    str
        Abbreviated month name (e.g., 'Jan', 'Feb').
    """
    names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return names[month_num - 1] if 1 <= month_num <= 12 else str(month_num)
