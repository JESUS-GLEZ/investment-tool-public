import pytest
from pydantic import ValidationError

from api import adapters
from api.schemas import ExecutionRequest


def test_fractional_holdings_and_zero_cash_reach_the_engine(tmp_path, monkeypatch):
    request = ExecutionRequest(
        execution_mode="investing",
        initial_cash=0,
        current_holdings=[
            {"symbol": "AAPL", "quantity": 2.5},
            {"symbol": "BTC/USD", "quantity": 0.003},
        ],
    )
    monkeypatch.setattr(adapters, "_OUTPUT_BASE", tmp_path)
    parameters = adapters.build_exec_params(request, "holdings")
    assert parameters.current_holdings == {"AAPL": 2.5, "BTC/USD": 0.003}
    assert parameters.cfg.model.initial_cash == 0
    assert request.model_dump(mode="json")["current_holdings"][1]["quantity"] == 0.003


@pytest.mark.parametrize(
    "holdings",
    [
        [{"symbol": "AAPL", "quantity": 0}],
        [{"symbol": "AAPL", "quantity": -1}],
        [{"symbol": "AAPL", "quantity": float("inf")}],
        [{"symbol": "AAPL", "quantity": float("nan")}],
        [{"symbol": "AAPL", "quantity": 1}, {"symbol": "AAPL", "quantity": 2}],
        [{"symbol": "NOT-IN-UNIVERSE", "quantity": 1}],
    ],
)
def test_invalid_holdings_are_rejected_before_enqueue(holdings):
    with pytest.raises(ValidationError):
        ExecutionRequest(execution_mode="investing", current_holdings=holdings)


def test_holdings_do_not_silently_change_backtesting_semantics():
    with pytest.raises(ValidationError, match="only to investment recommendations"):
        ExecutionRequest(current_holdings=[{"symbol": "AAPL", "quantity": 1}])
    with pytest.raises(ValidationError, match="Positive cash"):
        ExecutionRequest(execution_mode="investing", initial_cash=0)
    assert ExecutionRequest().current_holdings == []  # old requests remain valid
