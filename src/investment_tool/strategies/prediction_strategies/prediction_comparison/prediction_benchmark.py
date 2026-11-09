"""Prediction benchmark runner.

Evaluates multiple prediction strategies via **walk-forward**
(rolling-origin) evaluation: at each date *t*, the strategy sees only
data up to *t* and predicts returns for *t+1*.  The prediction is then
compared against the realised close-to-close return of the explicit target
period. Inputs obey the shared bar availability contract; all predictors are
scored on the same finite asset/period pairs.
"""

from __future__ import annotations

from investment_tool.output.policy import OutputMode

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from investment_tool.constants import Frequency, StockFields
from investment_tool.domain.market_time import observable_history, validate_bar_times
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.strategies.optimization_strategies.base_optimization_strategy import (
    ProgressCallback,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_metrics import (
    compute_all_prediction_metrics,
    compute_per_asset_metrics,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.prediction_plots import (
    plot_directional_accuracy_over_time,
    plot_error_distributions,
    plot_metric_bars,
    plot_per_asset_heatmap,
    plot_predictions_vs_actuals,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_config import (
    PredictorConfig,
)
from investment_tool.strategies.prediction_strategies.prediction_comparison.predictor_result import (
    PredictorResult,
)

logger = logging.getLogger(__name__)


class PredictionBenchmark:
    """Walk-forward benchmark for prediction strategy comparison.

    At each origin, completed historical bars available at the target's
    decision time are passed to the predictor. The target-period return
    is then scored. Calendar labels identify periods, not availability
    instants. Up to ``n_eval_periods`` completed targets are evaluated,
    leaving earlier data for warm-up / training.

    Parameters
    ----------
    stocks : list[Stock]
        Full stock entities (untruncated).
    sentiment_scores : dict[str, dict[Any, float]] | None
        Pre-computed sentiment scores.  ``None`` → empty per-stock dicts.
    n_eval_periods : int
        Number of walk-forward evaluation steps.
    min_history : int
        Minimum number of historical periods required before the first
        evaluation step.
    top_k : int
        Number of top assets for hit-rate metric computation.
    output_directory : Path | None
        Directory for comparison outputs.
    frequency : Frequency
        Declared calendar frequency from the repository (monthly by default).
    """

    def __init__(
        self,
        stocks: list[Stock],
        sentiment_scores: dict[str, dict[Any, float]] | None = None,
        n_eval_periods: int = 12,
        min_history: int = 12,
        top_k: int = 5,
        output_directory: Path | None = None,
        frequency: Frequency = Frequency.MONTHLY,
    ) -> None:
        if not stocks:
            raise ValueError("At least one stock is required.")
        if min(n_eval_periods, min_history, top_k) < 1:
            raise ValueError(
                "Evaluation periods, minimum history and top_k must be positive."
            )
        if len({stock.symbol for stock in stocks}) != len(stocks):
            raise ValueError("Stock symbols must be unique.")
        self.frequency = frequency
        self.stocks = [
            self._copy_stock(stock, validate_bar_times(stock.data, frequency))
            for stock in stocks
            if not stock.data.empty
        ]
        self._as_of = pd.Timestamp.now(tz="UTC")
        self._calendar = pd.DatetimeIndex([])
        self._targets: dict[pd.Timestamp, pd.Timestamp] = {}
        self._cutoffs: dict[pd.Timestamp, pd.Timestamp] = {}
        self._inputs: dict[pd.Timestamp, list[Stock]] = {}
        self._audit = pd.DataFrame()
        self.sentiment_scores: dict[str, dict[Any, float]] = (
            sentiment_scores
            if sentiment_scores is not None
            else {s.symbol: {} for s in stocks}
        )
        self.n_eval_periods = n_eval_periods
        self.min_history = min_history
        self.top_k = top_k
        self.output_directory = output_directory
        self._results: list[PredictorResult] = []

    @property
    def results(self) -> list[PredictorResult]:
        """Results from the last :meth:`run` call."""
        return self._results

    # ------------------------------------------------------------------
    # Walk-forward data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _copy_stock(stock: Stock, data: pd.DataFrame) -> Stock:
        # Preserve computed features when copying an already filtered input.
        copied = object.__new__(Stock)
        copied.symbol = stock.symbol
        copied.data = data.copy(deep=True)
        return copied

    def _build_evaluation_dates(self) -> list[pd.Timestamp]:
        """Choose explicit adjacent calendar periods, independently of stock order.

        min_history counts completed price bars (the first return needs an
        additional preceding close). n_eval_periods is a maximum, bounded by
        the available warm-up and completed targets.
        """
        labels = pd.DatetimeIndex(
            sorted({d for s in self.stocks for d in s.data.index})
        )
        if labels.empty:
            raise ValueError("Insufficient data for walk-forward evaluation")
        codes = {
            Frequency.MONTHLY: "M",
            Frequency.WEEKLY: "W-SUN",
            Frequency.DAILY: "D",
        }
        self._calendar = pd.period_range(
            labels[0], labels[-1], freq=codes[self.frequency]
        ).start_time
        completed_targets = {
            d
            for stock in self.stocks
            for d in stock.data.index[
                stock.data["is_complete"] & (stock.data["available_at"] <= self._as_of)
            ]
        }
        pairs = list(zip(self._calendar[:-1], self._calendar[1:]))
        self._targets = dict(pairs)
        dates = [
            origin
            for i, (origin, target) in enumerate(pairs)
            if i >= self.min_history - 1 and target in completed_targets
        ][-self.n_eval_periods :]
        if not dates:
            raise ValueError(
                f"Insufficient data for walk-forward evaluation: need "
                f"min_history={self.min_history} + at least 1 completed eval period."
            )
        zones = {
            zone
            for stock in self.stocks
            for zone in stock.data["market_timezone"].unique()
        }
        self._cutoffs = {
            origin: max(
                self._targets[origin].tz_localize(zone).tz_convert("UTC")
                for zone in zones
            )
            for origin in dates
        }
        self._inputs = {}
        return dates

    def _truncate_stocks(self, up_to: pd.Timestamp) -> list[Stock]:
        """Filter by actual availability and recompute features across calendar gaps.

        Eligibility depends solely on history available at decision time,
        never on whether a target return will later be available.
        """
        if up_to not in self._inputs:
            truncated = []
            calendar = self._calendar[self._calendar <= up_to]
            for stock in self.stocks:
                history = observable_history(
                    stock.data, self._targets[up_to], self._cutoffs[up_to]
                )
                if self.frequency == Frequency.DAILY:
                    # Daily equity bars are trading sessions; weekends/holidays
                    # are not artificial missing observations. Retain original
                    # session rows to mask bars unavailable at this decision.
                    history = history.reindex(
                        stock.data.index[stock.data.index <= up_to]
                    )
                else:
                    history = history.reindex(calendar)
                recent = history[StockFields.CLOSE].iloc[-self.min_history :]
                if (
                    len(recent) < self.min_history
                    or not (np.isfinite(recent) & (recent > 0)).all()
                ):
                    continue
                # Recompute returns/features after availability filtering so an
                # unavailable or absent close cannot leak through a stored return.
                truncated.append(Stock(stock.symbol, history))
            self._inputs[up_to] = truncated
        return [self._copy_stock(stock, stock.data) for stock in self._inputs[up_to]]

    @staticmethod
    def _truncate_sentiments(
        sentiment_scores: dict[str, dict[Any, float]],
        up_to: pd.Timestamp,
    ) -> dict[str, dict[Any, float]]:
        """Filter sentiment scores to dates ≤ *up_to*.

        Parameters
        ----------
        sentiment_scores : dict[str, dict[Any, float]]
            Full sentiment data.
        up_to : pd.Timestamp
            Cut-off date.

        Returns
        -------
        dict[str, dict[Any, float]]
            Filtered sentiment scores.
        """
        truncated: dict[str, dict[Any, float]] = {}
        for symbol, date_scores in sentiment_scores.items():
            truncated[symbol] = {
                d: s for d, s in date_scores.items() if pd.Timestamp(d) <= up_to
            }
        return truncated

    def _get_actual_returns(
        self, eval_date: pd.Timestamp, symbols: list[str]
    ) -> dict[str, float]:
        """Score only the named next calendar bar, never the next surviving row."""
        actuals: dict[str, float] = {}
        target = self._targets[eval_date]
        for stock in self.stocks:
            if stock.symbol not in symbols or target not in stock.data.index:
                continue
            row = stock.data.loc[target]
            if not row["is_complete"] or row["available_at"] > self._as_of:
                continue
            input_stock = next(
                s for s in self._inputs[eval_date] if s.symbol == stock.symbol
            )
            previous_close = input_stock.data[StockFields.CLOSE].iloc[-1]
            target_close = row[StockFields.CLOSE]
            if (
                np.isfinite(previous_close)
                and previous_close > 0
                and np.isfinite(target_close)
                and target_close > 0
            ):
                actuals[stock.symbol] = float(target_close / previous_close - 1)
        return actuals

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _run_single_predictor(
        self,
        config: PredictorConfig,
        eval_dates: list[pd.Timestamp],
    ) -> PredictorResult:
        """Run walk-forward evaluation for a single predictor.

        Parameters
        ----------
        config : PredictorConfig
            Predictor configuration.
        eval_dates : list[pd.Timestamp]
            Evaluation dates.

        Returns
        -------
        PredictorResult
            Complete result with predictions, actuals, and metrics.
        """
        all_predictions: list[dict[str, float]] = []
        all_actuals: list[dict[str, float]] = []
        failures: dict[str, str] = {}
        t_start = time.perf_counter()
        for eval_date in eval_dates:
            stocks = self._truncate_stocks(eval_date)
            symbols = [stock.symbol for stock in stocks]
            predictions: dict[str, float] = {}
            try:
                if stocks:
                    strategy = config.factory(
                        stocks,
                        self._truncate_sentiments(self.sentiment_scores, eval_date),
                    )
                    predictions = strategy.predict_returns()
            except Exception as exc:
                failures[self._targets[eval_date].isoformat()] = str(exc)
                logger.warning(
                    "Predictor '%s' failed at %s: %s", config.label, eval_date, exc
                )
            # Unknown assets are not allowed to expand the comparison universe.
            all_predictions.append(
                {symbol: predictions.get(symbol, float("nan")) for symbol in symbols}
            )
            all_actuals.append(self._get_actual_returns(eval_date, symbols))

        columns = sorted(stock.symbol for stock in self.stocks)
        targets = [self._targets[origin] for origin in eval_dates]
        return PredictorResult(
            config=config,
            predictions_df=pd.DataFrame(all_predictions, index=targets).reindex(
                columns=columns
            ),
            actuals_df=pd.DataFrame(all_actuals, index=targets).reindex(
                columns=columns
            ),
            wall_time_seconds=time.perf_counter() - t_start,
            evaluation_dates=[target.isoformat() for target in targets],
            decision_times=[self._cutoffs[origin].isoformat() for origin in eval_dates],
            failed_periods=failures,
        )

    def _score_common_sample(self, results: list[PredictorResult]) -> None:
        """Retain raw forecasts in one audit table; score the same pairs for all."""
        common = pd.DataFrame(
            True,
            index=results[0].predictions_df.index,
            columns=results[0].predictions_df.columns,
        )
        for result in results:
            common &= np.isfinite(result.predictions_df) & np.isfinite(
                result.actuals_df
            )
        scored_pairs = int(common.to_numpy().sum())
        scored_periods = int(common.any(axis=1).sum())
        audit: list[dict[str, Any]] = []
        origins = {target: origin for origin, target in self._targets.items()}
        for result in results:
            finite_predictions = np.isfinite(result.predictions_df)
            finite_actuals = np.isfinite(result.actuals_df)
            eligible_pairs = 0
            for i, target in enumerate(result.predictions_df.index):
                origin = origins[target]
                eligible = {stock.symbol for stock in self._inputs[origin]}
                eligible_pairs += len(eligible)
                for symbol in sorted(eligible):
                    audit.append(
                        {
                            "predictor": result.config.label,
                            "history_period": origin,
                            "target_period": target,
                            "decision_time": result.decision_times[i],
                            "symbol": symbol,
                            "prediction": result.predictions_df.loc[target, symbol],
                            "actual": result.actuals_df.loc[target, symbol],
                            "scored": bool(common.loc[target, symbol]),
                            "failure": result.failed_periods.get(
                                target.isoformat(), ""
                            ),
                        }
                    )
            result.metrics = compute_all_prediction_metrics(
                result.predictions_df.where(common),
                result.actuals_df.where(common),
                top_k=self.top_k,
            )
            result.metrics.update(
                {
                    "wall_time": result.wall_time_seconds,
                    "evaluation_periods": float(len(common)),
                    "history_bars": float(self.min_history),
                    "eligible_pairs": float(eligible_pairs),
                    "prediction_pairs": float(finite_predictions.to_numpy().sum()),
                    "actual_pairs": float(finite_actuals.to_numpy().sum()),
                    "scored_pairs": float(scored_pairs),
                    "scored_periods": float(scored_periods),
                    "failed_periods": float(len(result.failed_periods)),
                    "coverage": scored_pairs / eligible_pairs
                    if eligible_pairs
                    else 0.0,
                }
            )
            result.predictions_df = result.predictions_df.where(common)
            result.actuals_df = result.actuals_df.where(common)
            result.per_asset_metrics = compute_per_asset_metrics(
                result.predictions_df, result.actuals_df
            ).reindex(result.predictions_df.columns)
        self._audit = pd.DataFrame(
            audit,
            columns=[
                "predictor",
                "history_period",
                "target_period",
                "decision_time",
                "symbol",
                "prediction",
                "actual",
                "scored",
                "failure",
            ],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        predictors: list[PredictorConfig],
        progress_callback: ProgressCallback | None = None,
    ) -> list[PredictorResult]:
        """Run walk-forward evaluation for all predictors.

        Parameters
        ----------
        predictors : list[PredictorConfig]
            Predictor configurations to benchmark.
        progress_callback : ProgressCallback | None, optional
            Called after each predictor completes with
            ``(fraction_complete, message)``.

        Returns
        -------
        list[PredictorResult]
            One result per predictor, in the same order.

        Raises
        ------
        ValueError
            If *predictors* is empty or data is insufficient.
        """
        if not predictors:
            raise ValueError("At least one PredictorConfig is required.")

        if len({config.label for config in predictors}) != len(predictors):
            raise ValueError("Predictor labels must be unique.")
        eval_dates = self._build_evaluation_dates()
        logger.info(
            "Walk-forward evaluation: %d target periods from %s to %s",
            len(eval_dates),
            self._targets[eval_dates[0]].date(),
            self._targets[eval_dates[-1]].date(),
        )

        # Assemble shared causal inputs once, outside predictor timing. Each
        # predictor still gets independent copies so mutations cannot leak.
        for origin in eval_dates:
            self._truncate_stocks(origin)
        results: list[PredictorResult] = []
        n_predictors = len(predictors)
        for idx, config in enumerate(predictors):
            logger.info("Evaluating predictor '%s'", config.label)
            result = self._run_single_predictor(config, eval_dates)
            results.append(result)
            if progress_callback is not None:
                progress_callback(
                    (idx + 1) / n_predictors,
                    f"Predictor '{config.label}' complete "
                    f"({idx + 1}/{n_predictors})",
                )

        self._score_common_sample(results)
        self._results = results
        return results

    def summary_table(self) -> pd.DataFrame:
        """Build a DataFrame comparing all predictors side-by-side.

        Returns
        -------
        pd.DataFrame
            Rows = predictors, columns = metrics + wall_time.
        """
        rows: list[dict[str, Any]] = []
        for r in self._results:
            row: dict[str, Any] = {"predictor": r.config.label}
            row.update(r.metrics)
            row["wall_time_s"] = round(r.wall_time_seconds, 2)
            rows.append(row)
        return pd.DataFrame(rows).set_index("predictor")

    def save_comparison(
        self, output_mode: OutputMode = OutputMode.DIAGNOSTIC
    ) -> list[str]:
        """Save plots and CSV summary to ``output_directory``.

        Returns
        -------
        list[str]
            Paths to all saved files.
        """
        saved: list[str] = []

        # CSV summary
        if self._results:
            table = self.summary_table()
            out_dir = (
                Path(self.output_directory) / "prediction_comparison"
                if self.output_directory
                else Path.cwd() / "prediction_comparison"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "prediction_comparison_summary.csv"
            table.to_csv(csv_path)
            logger.info("Saved prediction comparison summary: %s", csv_path)
            saved.append(str(csv_path))
            audit_path = out_dir / "forecast_pairs.csv"
            self._audit.to_csv(audit_path, index=False)
            saved.append(str(audit_path))

            # Per-asset detail CSV
            for r in self._results:
                if not r.per_asset_metrics.empty:
                    asset_path = out_dir / f"per_asset_metrics_{r.config.label}.csv"
                    r.per_asset_metrics.to_csv(asset_path)
                    saved.append(str(asset_path))

        if not output_mode.save_chart_exports:
            return saved

        # Plots
        for plot_fn in [
            plot_predictions_vs_actuals,
            plot_error_distributions,
            plot_metric_bars,
            plot_directional_accuracy_over_time,
        ]:
            path = plot_fn(self._results, self.output_directory)
            if path:
                saved.append(path)

        path = plot_per_asset_heatmap(self._results, self.output_directory)
        if path:
            saved.append(path)

        return saved
