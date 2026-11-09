"""Recommendations value only observable history and never record real trades."""

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from investment_tool.app.recommendation import estimate_rebalance, initialize_holdings
from investment_tool.constants import Frequency
from investment_tool.domain.market_time import normalize_bars
from investment_tool.domain.model_data.entities.instance import Instance
from investment_tool.domain.model_data.entities.period import Period
from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.security_action import SecurityAction
from investment_tool.domain.model_data.entities.stock import Stock
from investment_tool.domain.model_data.model_data import ModelData
from investment_tool.domain.trading_costs import decision_trading_costs


@pytest.fixture()
def model():
    period = Period(
        pd.Timestamp("2024-04-01").to_pydatetime(),
        pd.DataFrame(),
        decision_time=pd.Timestamp("2024-04-01T04:00:00Z").to_pydatetime(),
    )
    result = ModelData()
    result.instance = Instance(
        date(2024, 1, 1), date(2024, 3, 31), Frequency.MONTHLY, period, None, period
    )
    result.periods = [period]
    result.news = {}
    stocks = []
    for symbol, price in [("SPY", 100), ("GLD", 50), ("BTC/USD", 50000)]:
        bars = normalize_bars(
            pd.DataFrame(
                {"close": [price, price, price * 100]},
                index=pd.date_range(
                    "2024-01-01", periods=3, freq="MS", tz="America/New_York"
                ),
            ),
            Frequency.MONTHLY,
            market_timezone="America/New_York",
            as_of="2024-06-01T00:00:00Z",
        )
        # The spectacular March close was not available at the decision.
        bars.loc[pd.Timestamp("2024-03-01"), "available_at"] = pd.Timestamp(
            "2024-04-15T00:00:00Z"
        )
        stocks.append(Stock(symbol, bars))
    result.stocks = stocks
    result.portfolio = Portfolio(result.instance, 100, transaction_cost_rate=0.01)
    return result


def test_rebalance_reconciles_holdings_cash_both_sided_fees_and_funding(model):
    initialize_holdings(model, {"SPY": 2.0})
    plan = estimate_rebalance(model, pd.Series({"SPY": 0.5, "GLD": 0.5}))
    rows = {row["symbol"]: row for row in plan["rows"]}
    assert plan["current_value"] == 300
    assert rows["SPY"]["reference_price"] == 100  # not the unavailable 10,000
    assert rows["SPY"]["price_period_end"] == "2024-02-29"
    assert rows["SPY"]["current_quantity"] == 2
    assert rows["SPY"]["trade_quantity"] == -0.5
    assert rows["SPY"]["action"] == "sell"
    assert rows["GLD"]["action"] == "buy"
    assert rows["GLD"]["estimated_quantity"] == pytest.approx(149.5 / 1.01 / 50)
    assert plan["estimated_cash"] == pytest.approx(0, abs=1e-12)
    assert plan["estimated_cost"] == pytest.approx(
        sum(abs(r["trade_value"]) * 0.01 for r in plan["rows"])
    )
    assert plan["estimated_value"] + plan["estimated_cost"] == pytest.approx(300)
    assert model.portfolio.positions == {"SPY": 2}
    assert model.portfolio.cash == 100
    assert model.portfolio.history == []
    assert model.instance.current_trading_period is model.periods[0]


def test_zero_cash_and_tiny_fractional_crypto_are_preserved(model):
    model.portfolio.cash = 0
    initialize_holdings(model, {"BTC/USD": 1e-7})
    plan = estimate_rebalance(model, pd.Series({"BTC/USD": 1.0}))
    assert plan["current_value"] == pytest.approx(0.005)
    assert plan["rows"][0]["estimated_quantity"] == pytest.approx(1e-7)
    assert plan["rows"][0]["action"] == "hold"
    assert plan["estimated_cost"] == 0


def test_deselected_asset_is_fully_sold_and_still_has_a_quote(model):
    initialize_holdings(model, {"SPY": 2})
    plan = estimate_rebalance(model, pd.Series({"GLD": 1.0}))
    sold = next(row for row in plan["rows"] if row["symbol"] == "SPY")
    assert sold["estimated_quantity"] == 0
    assert sold["trade_quantity"] == -2
    assert sold["estimated_cost"] == 2


@pytest.mark.parametrize("problem", ["absent", "unavailable", "invalid"])
def test_unpriced_holdings_are_not_dropped_or_valued_at_zero(model, problem):
    stock = model.stocks[0]
    if problem == "absent":
        model.stocks = model.stocks[1:]
    elif problem == "unavailable":
        stock.data["available_at"] = pd.Timestamp("2025-01-01T00:00:00Z")
    else:
        stock.data.loc[pd.Timestamp("2024-02-01"), "close"] = np.nan
    with pytest.raises(ValueError, match="SPY"):
        initialize_holdings(model, {"SPY": 2})
    assert model.portfolio.positions == {}


def test_past_entitlements_are_not_reapplied_to_imported_units(model):
    event = SecurityAction(
        "SPY",
        pd.Timestamp("2024-02-01T00:00:00Z").to_pydatetime(),
        pd.Timestamp("2024-01-31T00:00:00Z").to_pydatetime(),
        10,
        "Synthetic test event",
    )
    model.security_actions = (event,)
    initialize_holdings(model, {"SPY": 2})
    costs = decision_trading_costs(
        model.portfolio,
        model.stocks,
        model.instance.current_trading_period,
        pd.Index(["SPY", "GLD"]),
        model.corporate_action_events,
    )
    assert costs is not None
    assert costs.current_weights[0] == pytest.approx(2 / 3)
    model.portfolio.apply_security_actions(
        model.corporate_action_events,
        model.instance.current_trading_period.decision_time,
    )
    assert model.portfolio.positions == {"SPY": 2}
    assert model.portfolio.cash == 100


@pytest.mark.parametrize(
    "weights", [{"SPY": 0.8}, {"SPY": -0.1, "GLD": 1.1}, {"SPY": np.nan}]
)
def test_invalid_targets_cannot_produce_a_plausible_plan(model, weights):
    with pytest.raises(ValueError, match="weights"):
        estimate_rebalance(model, pd.Series(weights))


def test_execution_binds_imported_holdings_before_cost_aware_selection(
    model, tmp_path, monkeypatch
):
    from investment_tool.app import execution_coordinator as coordinator
    from investment_tool.config import AppConfig
    from investment_tool.constants import DataSource, ExecutionMode, StrategyType
    from investment_tool.output.output_manager import OutputManager
    from investment_tool.output.policy import OutputMode

    monkeypatch.setattr(
        coordinator.ModelDataFactory, "create_model_data", lambda **kwargs: model
    )
    monkeypatch.setattr(coordinator, "write_run_metadata", lambda *_: None)
    monkeypatch.setattr(coordinator, "record_input_snapshot", lambda *_: None)
    bound = []

    def select(**kwargs):
        assert bound[0].positions == {"SPY": 2}
        costs = decision_trading_costs(
            bound[0],
            model.stocks,
            model.instance.current_trading_period,
            pd.Index(["SPY", "GLD"]),
        )
        assert costs.current_weights[0] == pytest.approx(2 / 3)
        return pd.Series({"SPY": 1.0})

    strategy = SimpleNamespace(
        stocks=model.stocks,
        bind_portfolio=lambda portfolio, events: bound.append(portfolio),
        run_strategy=select,
        output_mode=OutputMode.STANDARD,
        last_optimization_diagnostics={},
    )
    monkeypatch.setattr(coordinator, "_build_strategy", lambda **kwargs: strategy)
    monkeypatch.setattr(
        OutputManager, "generate_live_output", lambda *args, **kwargs: {}
    )
    result = coordinator.execute(
        coordinator.Parameters(
            data_source=DataSource.LOCAL,
            execution_mode=ExecutionMode.INVESTING,
            frequency=Frequency.MONTHLY,
            output_directory=tmp_path,
            strategy_type=StrategyType.OPTIMIZATION,
            current_holdings={"SPY": 2},
            cfg=AppConfig(asset_config_file="example_asset_universe.json"),
        )
    )
    assert result["portfolio"] is model.portfolio
    assert model.portfolio.positions == {"SPY": 2}
    assert result["rebalance_plan"]["current_value"] == 300
