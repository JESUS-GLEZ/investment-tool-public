"""Cost forecasts use actual holdings and strictly observable decision prices."""

from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from investment_tool.constants import Frequency
from investment_tool.domain.market_time import normalize_bars
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.entities.cash_settlement import CashSettlement
from investment_tool.domain.trading_costs import (
    TradingCosts,
    decision_trading_costs,
    expected_net_return,
    net_return_gradient,
)


def state(rate=0.001):
    period = Period(
        datetime(2025, 3, 1),
        pd.DataFrame(),
        decision_time=datetime.fromisoformat("2025-03-01T05:00:00Z"),
    )
    instance = Instance(
        date(2025, 1, 1), date(2025, 3, 31), Frequency.MONTHLY, period, None, period
    )
    portfolio = Portfolio(instance, 0.0, rate)
    portfolio.positions = {"A": 5.0, "B": 5.0}
    stocks = []
    for symbol, closes in [("A", [100, 200, 1e9]), ("B", [100, 50, 1e9])]:
        frame = pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": 1000,
            },
            index=pd.date_range(
                "2025-01-01", periods=3, freq="MS", tz="America/New_York"
            ),
        )
        stocks.append(
            Stock(
                symbol,
                normalize_bars(
                    frame,
                    Frequency.MONTHLY,
                    market_timezone="America/New_York",
                    as_of="2026-01-01T00:00:00Z",
                ),
            )
        )
    return portfolio, stocks, period


def test_actual_holdings_drift_and_asset_order_not_old_target_weights():
    portfolio, stocks, period = state()
    costs = decision_trading_costs(portfolio, stocks, period, pd.Index(["B", "A"]))
    np.testing.assert_allclose(costs.current_weights, [0.2, 0.8])
    assert costs.fraction(np.array([0.5, 0.5])) == pytest.approx(0.0006)
    for stock in stocks:
        stock.data.loc["2025-03-01", ["open", "close"]] = 1
    again = decision_trading_costs(portfolio, stocks, period, pd.Index(["B", "A"]))
    assert costs.record() == again.record()
    assert portfolio.positions == {"A": 5, "B": 5} and portfolio.cash == 0


def test_outside_holdings_are_priced_and_charged_for_exit():
    portfolio, stocks, period = state()
    costs = decision_trading_costs(portfolio, stocks, period, pd.Index(["A"]))
    np.testing.assert_allclose(costs.current_weights, [0.8])
    assert costs.outside_weight == pytest.approx(0.2)
    assert costs.fraction(np.ones(1)) == pytest.approx(0.0004)
    restricted = TradingCosts(0.001, np.array([0.8, 0.2])).restrict(np.array([0]))
    assert restricted.outside_weight == pytest.approx(costs.outside_weight)
    assert restricted.fraction(np.ones(1)) == pytest.approx(costs.fraction(np.ones(1)))


def test_cash_entry_and_zero_cost_compatibility():
    portfolio, stocks, period = state()
    portfolio.positions = {}
    portfolio.cash = 1000
    costs = decision_trading_costs(portfolio, stocks, period, pd.Index(["A", "B"]))
    assert costs.fraction(np.array([0.3, 0.7])) == 0.001
    portfolio.transaction_cost_rate = 0
    assert decision_trading_costs(portfolio, [], None, pd.Index(["MISSING"])) is None


def test_late_history_and_missing_held_quotes_never_use_future_open():
    portfolio, stocks, period = state()
    stocks[0].data.loc[:"2025-02-01", "available_at"] = pd.Timestamp(
        "2025-03-02T05:00:00Z"
    )
    with pytest.raises(ValueError, match="No decision-time price for held asset A"):
        decision_trading_costs(portfolio, stocks, period, pd.Index(["B"]))
    with pytest.raises(ValueError, match="No decision-time history for held asset A"):
        decision_trading_costs(portfolio, stocks[1:], period, pd.Index(["B"]))


@pytest.mark.parametrize("known", [False, True])
def test_only_already_known_redemptions_change_estimated_costs(known):
    portfolio, stocks, period = state()
    published = "2025-02-28T21:00:00Z" if known else "2025-03-01T06:00:00Z"
    event = CashSettlement(
        "B",
        datetime.fromisoformat("2025-02-28T20:00:00Z"),
        datetime.fromisoformat(published),
        50,
        "reviewed cash redemption",
    )
    costs = decision_trading_costs(portfolio, stocks, period, pd.Index(["A"]), [event])
    assert costs.outside_weight == pytest.approx(0 if known else 0.2)
    assert costs.fraction(np.ones(1)) == pytest.approx(0.0002 if known else 0.0004)
    assert portfolio.positions == {"A": 5, "B": 5}
    assert portfolio.cash == 0 and portfolio.corporate_actions == []


def test_net_return_and_gradient_share_same_fee_definition():
    mu = np.array([0.03, 0.05, 0.04])
    costs = TradingCosts(0.01, np.array([0.4, 0.1, 0.2]), 0.1)
    weights = np.array([0.3, 0.5, 0.2])
    assert expected_net_return(mu, weights, costs) == pytest.approx(
        mu @ weights - 0.006
    )
    numeric = []
    for i in range(2):  # Third coordinate is at an L1 kink.
        delta = np.eye(3)[i] * 1e-7
        numeric.append(
            (
                expected_net_return(mu, weights + delta, costs)
                - expected_net_return(mu, weights - delta, costs)
            )
            / 2e-7
        )
    np.testing.assert_allclose(
        net_return_gradient(mu, weights, costs)[:2], numeric, atol=1e-9
    )
    assert expected_net_return(mu, weights) == float(mu @ weights)


@pytest.mark.parametrize(
    "rate,weights,outside",
    [
        (-1, [0], 0),
        (float("nan"), [0], 0),
        (0.1, [-0.1], 0),
        (0.1, [1], 0.1),
        (0.1, [float("inf")], 0),
    ],
)
def test_cost_state_requires_finite_unlevered_holdings(rate, weights, outside):
    with pytest.raises(ValueError, match="Invalid decision-time"):
        TradingCosts(rate, np.array(weights), outside)
