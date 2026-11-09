"""Mandatory entitlements preserve holdings and cannot be credited twice."""

from datetime import datetime
from dataclasses import replace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from investment_tool.domain.model_data.entities.portfolio import Portfolio
from investment_tool.domain.model_data.entities.security_action import SecurityAction


def action(*, retain=False, published="2020-04-03T12:00:00Z"):
    return SecurityAction(
        "OLD",
        datetime.fromisoformat("2020-04-03T12:00:00Z"),
        datetime.fromisoformat(published),
        5,
        "reviewed issuer notice",
        (("NEW", 2), ("SPIN", 0.5)),
        retain,
    )


def test_mixed_merger_accumulates_existing_successor_and_rebalances():
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3, "NEW": 4}
    portfolio.apply_security_actions(
        [action()], datetime.fromisoformat("2020-04-04T00:00:00Z")
    )
    assert portfolio.cash == 25
    assert portfolio.positions == {"NEW": 10, "SPIN": 1.5}
    assert (
        portfolio.calculate_portfolio_value(pd.Series({"NEW": 20, "SPIN": 10})) == 240
    )
    portfolio.update(
        pd.Series({"LIVE": 1.0}),
        execution_prices=pd.Series({"NEW": 20, "SPIN": 10, "LIVE": 30}),
    )
    assert portfolio.positions == {"LIVE": 8}
    assert portfolio.cash == pytest.approx(0)


def test_spin_off_keeps_parent_and_never_repeats_after_new_purchase():
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3}
    event = action(retain=True)
    cutoff = datetime.fromisoformat("2020-04-04T00:00:00Z")
    portfolio.apply_security_actions([event], cutoff)
    assert portfolio.positions == {"OLD": 3, "NEW": 6, "SPIN": 1.5}
    portfolio.positions["OLD"] = 10
    portfolio.apply_security_actions([event], cutoff)
    assert portfolio.positions == {"OLD": 10, "NEW": 6, "SPIN": 1.5}
    assert portfolio.cash == 25
    assert len(portfolio.corporate_actions) == 1


def test_unheld_distribution_does_not_apply_to_later_purchase():
    portfolio = Portfolio(MagicMock(), 10)
    cutoff = datetime.fromisoformat("2020-04-04T00:00:00Z")
    portfolio.apply_security_actions([action(retain=True)], cutoff)
    portfolio.positions = {"OLD": 2}
    portfolio.apply_security_actions([action(retain=True)], cutoff)
    assert portfolio.positions == {"OLD": 2}
    assert portfolio.cash == 10
    assert portfolio.corporate_actions == []


def test_publication_and_effectiveness_are_both_required():
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3}
    event = action(published="2020-04-04T15:00:00Z")
    portfolio.apply_security_actions(
        [event], datetime.fromisoformat("2020-04-04T14:30:00Z")
    )
    assert portfolio.positions == {"OLD": 3}
    portfolio.apply_security_actions(
        [event], datetime.fromisoformat("2020-04-04T15:00:00Z")
    )
    assert portfolio.positions == {"NEW": 6, "SPIN": 1.5}


@pytest.mark.parametrize("retain", [False, True])
def test_incomplete_action_preserves_all_state_and_can_be_repaired(retain):
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3, "NEW": 4}
    event = replace(action(retain=retain), unresolved_entitlements=("Warrant quotes",))
    cutoff = datetime.fromisoformat("2020-04-04T00:00:00Z")
    with pytest.raises(
        ValueError, match="Incomplete security action for OLD: Warrant quotes"
    ):
        portfolio.apply_security_actions([event], cutoff)
    assert portfolio.positions == {"OLD": 3, "NEW": 4}
    assert portfolio.cash == 10
    assert portfolio.corporate_actions == []
    assert portfolio._applied_actions == set()

    # Replacing the same event after review must not be lost as already applied.
    portfolio.apply_security_actions(
        [replace(event, unresolved_entitlements=())], cutoff
    )
    assert portfolio.cash == 25
    assert portfolio.positions == ({"OLD": 3} if retain else {}) | {
        "NEW": 10,
        "SPIN": 1.5,
    }
    assert len(portfolio.corporate_actions) == 1


def test_incomplete_unheld_action_cannot_credit_a_later_purchase():
    portfolio = Portfolio(MagicMock(), 10)
    event = replace(action(retain=True), unresolved_entitlements=("Warrant quotes",))
    cutoff = datetime.fromisoformat("2020-04-04T00:00:00Z")
    portfolio.apply_security_actions([event], cutoff)
    portfolio.positions = {"OLD": 2}
    portfolio.apply_security_actions(
        [replace(event, unresolved_entitlements=())], cutoff
    )
    assert portfolio.positions == {"OLD": 2}
    assert portfolio.cash == 10
    assert portfolio.corporate_actions == []


def test_incomplete_action_is_not_processed_before_publication():
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3}
    event = replace(
        action(published="2020-04-04T15:00:00Z"),
        unresolved_entitlements=("Warrant quotes",),
    )
    portfolio.apply_security_actions(
        [event], datetime.fromisoformat("2020-04-04T14:30:00Z")
    )
    assert portfolio.positions == {"OLD": 3}
    assert portfolio._applied_actions == set()
    with pytest.raises(ValueError, match="Incomplete security action"):
        portfolio.apply_security_actions(
            [event], datetime.fromisoformat("2020-04-04T15:00:00Z")
        )


def test_incomplete_successor_prevents_every_credit_in_the_action_batch():
    portfolio = Portfolio(MagicMock(), 10)
    portfolio.positions = {"OLD": 3, "NEW": 4}
    complete = action()
    late = datetime.fromisoformat("2020-04-04T00:00:00Z")
    incomplete = SecurityAction(
        "SPIN",
        late,
        late,
        0,
        "reviewed notice",
        retain_original=True,
        unresolved_entitlements=("Warrant quotes",),
    )
    with pytest.raises(ValueError, match="Incomplete security action for SPIN"):
        portfolio.apply_security_actions([incomplete, complete], late)
    assert portfolio.positions == {"OLD": 3, "NEW": 4}
    assert portfolio.cash == 10
    assert portfolio.history == []
    assert portfolio.corporate_actions == []
    assert portfolio._applied_actions == set()
