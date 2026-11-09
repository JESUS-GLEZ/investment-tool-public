"""Analytical performance cases shared by execution, comparison and reporting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.output.export import DataExporter
from investment_tool.output.metrics import MetricsCalculator
from investment_tool.output.report import ReportBuilder
from investment_tool.output.strategy_comparator import StrategyComparator


def snapshots(returns, start="2024-01-01", frequency=Frequency.MONTHLY):
    codes = {Frequency.MONTHLY: "MS", Frequency.WEEKLY: "W-MON", Frequency.DAILY: "D"}
    labels = pd.date_range(start, periods=len(returns), freq=codes[frequency])
    delta = {
        Frequency.MONTHLY: pd.offsets.MonthBegin(),
        Frequency.WEEKLY: pd.Timedelta(days=7),
        Frequency.DAILY: pd.Timedelta(days=1),
    }[frequency]
    values = 1000 * np.cumprod(1 + np.asarray(returns))
    return [
        {
            "date": label + delta - pd.Timedelta(days=1),
            "period_start": label.date().isoformat(),
            "period_end": (label + delta - pd.Timedelta(days=1)).date().isoformat(),
            "date_basis": "period_end_label",
            "starting_value": 1000.0,
            "total_value": float(value),
            "return": float(ret),
        }
        for label, value, ret in zip(labels, values, returns)
    ]


def test_compounding_loss_is_not_compounded_arithmetic_mean():
    rows = snapshots([0.1, -0.1])
    metrics = MetricsCalculator(rows, [], Frequency.MONTHLY).compute_all()
    assert metrics["initial_value"] == pytest.approx(1000)
    assert metrics["final_value"] == pytest.approx(990)
    assert metrics["total_return"] == pytest.approx(-0.01)
    assert metrics["average_return"] == pytest.approx(0)
    assert metrics["annualized_return"] == pytest.approx(0.99 ** (365.25 / 60) - 1)
    assert metrics["sharpe_ratio"] == pytest.approx(0)
    assert metrics["metric_definitions"]["annualization_basis"] == "calendar_act_365_25"


@pytest.mark.parametrize("first", [0.2, -0.2])
def test_total_return_includes_first_period_even_for_one_snapshot(first):
    metrics = MetricsCalculator(snapshots([first]), [], Frequency.MONTHLY).compute_all()
    assert metrics["total_return"] == pytest.approx(first)
    assert metrics["initial_value"] == pytest.approx(1000)
    assert np.isnan(metrics["sharpe_ratio"])
    assert metrics["max_drawdown"] == pytest.approx(min(first, 0))


@pytest.mark.parametrize(
    "frequency,ppy",
    [(Frequency.MONTHLY, 12), (Frequency.WEEKLY, 52), (Frequency.DAILY, 252)],
)
@pytest.mark.parametrize("risk_free", [0.0, 0.06])
def test_sharpe_uses_arithmetic_excess_with_sample_deviation(frequency, ppy, risk_free):
    returns = np.array([0.01, -0.02, 0.03, 0.015, -0.005, 0.02])
    metrics = MetricsCalculator(
        snapshots(returns, frequency=frequency), [], frequency, risk_free
    ).compute_all()
    excess = returns - ((1 + risk_free) ** (1 / ppy) - 1)
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(ppy)
    assert metrics["sharpe_ratio"] == pytest.approx(expected)
    assert metrics["annualized_volatility"] == pytest.approx(
        returns.std(ddof=1) * np.sqrt(ppy)
    )
    assert metrics["rolling_sharpe"].iloc[-1] == pytest.approx(expected)
    downside = np.sqrt(np.mean(np.minimum(excess, 0) ** 2))
    assert metrics["sortino_ratio"] == pytest.approx(
        excess.mean() / downside * np.sqrt(ppy)
    )


def test_cagr_spans_leap_year_and_includes_final_period():
    rows = snapshots([0.01] * 12)
    metrics = MetricsCalculator(rows, [], Frequency.MONTHLY).compute_performance()
    assert metrics["annualized_return"] == pytest.approx(
        (1.01**12) ** (365.25 / 366) - 1
    )
    assert metrics["total_return"] == pytest.approx(1.01**12 - 1)
    assert np.isnan(metrics["sharpe_ratio"])
    assert np.isnan(metrics["sortino_ratio"])
    assert np.isnan(metrics["calmar_ratio"])


def test_single_and_comparison_definitions_match_exactly(tmp_path):
    rows = snapshots([0.1, -0.05, 0.03, -0.01, 0.02, 0.015])
    metrics = MetricsCalculator(rows, [], Frequency.MONTHLY, 0.04).compute_all()
    comparator = StrategyComparator(
        {"strategy": rows}, Frequency.MONTHLY, tmp_path, 0.04
    )
    compared = comparator._compute_comparison_metrics()["strategy"]
    for key in [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "calmar_ratio",
        "max_drawdown",
    ]:
        assert compared[key] == pytest.approx(metrics[key])
    pd.testing.assert_series_equal(
        comparator._calculators["strategy"].rolling_sharpe(6), metrics["rolling_sharpe"]
    )


def test_starting_loss_is_drawdown_from_original_capital():
    metrics = MetricsCalculator(
        snapshots([-0.2, 0.1]), [], Frequency.MONTHLY
    ).compute_all()
    assert metrics["drawdown_series"].tolist() == pytest.approx([-0.2, -0.12])
    assert metrics["max_drawdown"] == pytest.approx(-0.2)
    assert metrics["max_drawdown_start"] == "2024-01-01"
    assert metrics["max_drawdown_duration_periods"] == 2
    assert metrics["calmar_ratio"] == pytest.approx(metrics["annualized_return"] / 0.2)


def test_legacy_results_use_geometric_period_count_without_guessing_dates():
    rows = snapshots([0.1, -0.1])
    for row in rows:
        for key in ["period_start", "period_end", "date_basis", "starting_value"]:
            del row[key]
    metrics = MetricsCalculator(rows, [], Frequency.MONTHLY).compute_all()
    assert metrics["annualized_return"] == pytest.approx(0.99**6 - 1)
    assert metrics["initial_value"] == pytest.approx(1000)
    assert (
        metrics["metric_definitions"]["annualized_return"] == "geometric_period_return"
    )


def test_complete_loss_has_minus_one_growth_with_explicit_starting_capital():
    metrics = MetricsCalculator(
        snapshots([0.1, -1.0]), [], Frequency.MONTHLY
    ).compute_all()
    assert metrics["final_value"] == 0
    assert metrics["total_return"] == -1
    assert metrics["annualized_return"] == -1
    assert metrics["max_drawdown"] == -1


def test_zero_volatility_exports_valid_json_and_report_dash(tmp_path):
    metrics = MetricsCalculator(
        snapshots([0.01] * 6), [], Frequency.MONTHLY
    ).compute_all()
    path = DataExporter(tmp_path).export_metrics_json(metrics)[0]
    parsed = json.loads(
        Path(path).read_text(),
        parse_constant=lambda value: pytest.fail(f"Nonstandard JSON number {value}"),
    )
    assert parsed["sharpe_ratio"] is None
    assert parsed["metric_definitions"]["version"] == 2
    assert "CAGR" in ReportBuilder._section_summary(metrics)
    assert "nan" not in ReportBuilder._section_summary(metrics)


def test_inconsistent_portfolio_values_are_rejected():
    rows = snapshots([0.1, 0.2])
    rows[1]["total_value"] = 500
    with pytest.raises(ValueError, match="reconcile"):
        MetricsCalculator(rows, [], Frequency.MONTHLY).compute_performance()


def test_daily_cagr_includes_weekend_without_inventing_trading_observations():
    rows = snapshots([0.01, 0.02], start="2024-03-08", frequency=Frequency.DAILY)
    rows[1].update(
        date=pd.Timestamp("2024-03-11"),
        period_start="2024-03-11",
        period_end="2024-03-11",
    )
    metrics = MetricsCalculator(rows, [], Frequency.DAILY).compute_performance()
    assert metrics["annualized_return"] == pytest.approx(
        (1.01 * 1.02) ** (365.25 / 4) - 1
    )
    assert metrics["total_periods"] == 2


def test_empty_results_are_unavailable_instead_of_zero_performance():
    metrics = MetricsCalculator([], [], Frequency.MONTHLY).compute_all()
    assert np.isnan(metrics["total_return"])
    assert np.isnan(metrics["annualized_return"])
    assert np.isnan(metrics["sharpe_ratio"])
    assert metrics["total_periods"] == 0
