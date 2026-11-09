"""Tests for MetricsCalculator."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.output.metrics import MetricsCalculator


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_results(
    n: int = 12,
    start_value: float = 10000.0,
    monthly_return: float = 0.01,
) -> list[dict[str, Any]]:
    """Build synthetic per-period results."""
    results: list[dict[str, Any]] = []
    value = start_value
    base = datetime(2024, 1, 1)
    for i in range(n):
        ret = monthly_return if i > 0 else 0.0
        value *= 1 + ret
        results.append(
            {
                "date": datetime(base.year, base.month + i, 15)
                if base.month + i <= 12
                else datetime(base.year + 1, (base.month + i - 1) % 12 + 1, 15),
                "total_value": value,
                "return": ret,
            }
        )
    return results


def _make_portfolio_history(n: int = 12) -> list[dict[str, Any]]:
    """Build synthetic portfolio history entries."""
    history: list[dict[str, Any]] = []
    for i in range(n):
        history.append(
            {
                "date": datetime(2024, 1 + i if i < 12 else 1, 15),
                "cash": 500.0,
                "positions": {"AAPL": 10.0, "GOOG": 5.0, "MSFT": 8.0},
                "trades": {"AAPL": 1.0, "GOOG": -0.5},
                "prices": {"AAPL": 150.0, "GOOG": 200.0, "MSFT": 100.0},
                "pre_rebalance_value": 10000.0 + i * 100,
                "post_rebalance_value": 10050.0 + i * 100,
            }
        )
    return history


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    return _make_results()


@pytest.fixture
def sample_history() -> list[dict[str, Any]]:
    return _make_portfolio_history()


@pytest.fixture
def calculator(
    sample_results: list[dict[str, Any]],
    sample_history: list[dict[str, Any]],
) -> MetricsCalculator:
    return MetricsCalculator(
        results=sample_results,
        portfolio_history=sample_history,
        frequency=Frequency.MONTHLY,
        risk_free_rate=0.0,
    )


# ── Tests ───────────────────────────────────────────────────────────────────


class TestComputeAll:
    """Test compute_all returns the expected keys and types."""

    def test_returns_dict(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert isinstance(metrics, dict)

    def test_contains_basic_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        for key in [
            "average_return",
            "std_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_contains_drawdown_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        for key in [
            "max_drawdown",
            "max_drawdown_duration_periods",
            "drawdown_series",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_contains_risk_adjusted_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert "calmar_ratio" in metrics
        assert "sortino_ratio" in metrics

    def test_contains_win_loss_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert "win_rate" in metrics
        assert "profit_factor" in metrics

    def test_contains_rolling_sharpe(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert "rolling_sharpe" in metrics
        assert isinstance(metrics["rolling_sharpe"], pd.Series)

    def test_contains_breakdown_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert "monthly_returns" in metrics
        assert "yearly_returns" in metrics

    def test_contains_portfolio_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        for key in [
            "turnover_per_period",
            "average_turnover",
            "total_turnover",
            "weight_history",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_contains_summary_keys(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        for key in [
            "total_return",
            "total_periods",
            "initial_value",
            "final_value",
            "final_hhi",
            "final_top3_weight",
        ]:
            assert key in metrics, f"Missing key: {key}"


class TestBasicMetrics:
    """Test basic return and volatility calculations."""

    def test_positive_returns_give_positive_annualized(self) -> None:
        results = _make_results(n=12, monthly_return=0.02)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["annualized_return"] > 0

    def test_negative_returns_give_negative_annualized(self) -> None:
        results = _make_results(n=12, monthly_return=-0.02)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["annualized_return"] < 0

    def test_zero_returns(self) -> None:
        results = _make_results(n=12, monthly_return=0.0)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["average_return"] == pytest.approx(0.0)

    def test_sharpe_ratio_sign_matches_return(self) -> None:
        results = _make_results(n=12, monthly_return=0.01)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["sharpe_ratio"] > 0


class TestDrawdownMetrics:
    """Test drawdown computation."""

    def test_constant_growth_no_drawdown(self) -> None:
        results = _make_results(n=12, monthly_return=0.01)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["max_drawdown"] == pytest.approx(0.0)

    def test_drawdown_with_loss(self) -> None:
        """Inject a loss to verify drawdown is negative."""
        results = _make_results(n=6, monthly_return=0.02)
        # Inject a loss at period 3
        results[3]["return"] = -0.10
        results[3]["total_value"] = results[2]["total_value"] * 0.90
        for i in range(4, 6):
            results[i]["total_value"] = results[i - 1]["total_value"] * 1.02
            results[i]["return"] = 0.02

        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["max_drawdown"] < 0

    def test_drawdown_series_is_series(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        assert isinstance(metrics["drawdown_series"], pd.Series)


class TestWinLossMetrics:
    """Test win rate and profit factor."""

    def test_all_positive_returns_win_rate_100(self) -> None:
        # First period has 0 return, rest have positive
        results = _make_results(n=12, monthly_return=0.01)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        # period 0 has return=0 which is neither positive nor negative
        assert metrics["win_rate"] > 0.8

    def test_profit_factor_positive_for_profitable(self) -> None:
        results = _make_results(n=12, monthly_return=0.01)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["profit_factor"] > 0


class TestPortfolioMetrics:
    """Test turnover and weight history computation."""

    def test_turnover_computed(
        self,
        sample_results: list[dict[str, Any]],
        sample_history: list[dict[str, Any]],
    ) -> None:
        calc = MetricsCalculator(sample_results, sample_history, Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert len(metrics["turnover_per_period"]) == len(sample_history)

    def test_weight_history_is_dataframe(
        self,
        sample_results: list[dict[str, Any]],
        sample_history: list[dict[str, Any]],
    ) -> None:
        calc = MetricsCalculator(sample_results, sample_history, Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert isinstance(metrics["weight_history"], pd.DataFrame)

    def test_empty_portfolio_history(self) -> None:
        results = _make_results(n=6)
        calc = MetricsCalculator(results, [], Frequency.MONTHLY)
        metrics = calc.compute_all()
        assert metrics["average_turnover"] == 0.0
        assert metrics["total_transaction_costs"] is None

    def test_total_fees_require_complete_recorded_evidence(self):
        history = _make_portfolio_history(3)
        calculator = MetricsCalculator(_make_results(3), history, Frequency.MONTHLY)
        assert calculator.compute_all()["total_transaction_costs"] is None
        for entry, fee in zip(history, [1.25, 0.0, 2.5]):
            entry["transaction_costs"] = fee
        assert calculator.compute_all()["total_transaction_costs"] == 3.75
        history[0]["transaction_costs"] = None
        assert calculator.compute_all()["total_transaction_costs"] is None

    @pytest.mark.parametrize("bad", [-1, float("inf"), float("nan"), True, "1"])
    def test_invalid_fee_records_are_unavailable(self, bad):
        history = _make_portfolio_history(1)
        history[0]["transaction_costs"] = bad
        assert (
            MetricsCalculator(
                _make_results(1), history, Frequency.MONTHLY
            ).compute_all()["total_transaction_costs"]
            is None
        )


class TestWeightConcentration:
    """Test static weight concentration helper."""

    def test_equal_weights(self) -> None:
        w = pd.Series([0.25, 0.25, 0.25, 0.25], index=["A", "B", "C", "D"])
        result = MetricsCalculator.compute_weight_concentration(w)
        assert result["hhi"] == pytest.approx(0.25)
        assert result["num_active_positions"] == 4
        assert result["effective_n"] == pytest.approx(4.0)

    def test_concentrated_portfolio(self) -> None:
        w = pd.Series([1.0, 0.0, 0.0], index=["A", "B", "C"])
        result = MetricsCalculator.compute_weight_concentration(w)
        assert result["hhi"] == pytest.approx(1.0)
        assert result["num_active_positions"] == 1
        assert result["top3_weight"] == pytest.approx(1.0)

    def test_zero_weights(self) -> None:
        w = pd.Series([0.0, 0.0, 0.0], index=["A", "B", "C"])
        result = MetricsCalculator.compute_weight_concentration(w)
        assert result["num_active_positions"] == 0


class TestReturnBreakdowns:
    """Test monthly/yearly return breakdowns."""

    def test_monthly_returns_shape(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        mr = metrics["monthly_returns"]
        assert isinstance(mr, pd.DataFrame)
        assert not mr.empty

    def test_yearly_returns_is_series(self, calculator: MetricsCalculator) -> None:
        metrics = calculator.compute_all()
        yr = metrics["yearly_returns"]
        assert isinstance(yr, pd.Series)


class TestDifferentFrequencies:
    """Test that metrics compute correctly across frequencies."""

    @pytest.mark.parametrize(
        "freq", [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY]
    )
    def test_basic_metrics_compute(self, freq: Frequency) -> None:
        results = _make_results(n=24, monthly_return=0.005)
        calc = MetricsCalculator(results, [], freq)
        metrics = calc.compute_all()
        assert "annualized_return" in metrics
        assert "sharpe_ratio" in metrics
