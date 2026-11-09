"""Submission failures must not leave apparently queued jobs in the dashboard."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from api.db_store import DBRunStore
from api.models import Base, UserRow
from api.run_submission import submit_run
from api.schemas import (
    ExecutionRequest,
    PredictionComparisonRequest,
    RunType,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)


@pytest.mark.parametrize(
    "run_type,request_type",
    [
        (RunType.EXECUTION, ExecutionRequest),
        (RunType.STRATEGY_COMPARISON, StrategyComparisonRequest),
        (RunType.SOLVER_COMPARISON, SolverComparisonRequest),
        (RunType.PREDICTION_COMPARISON, PredictionComparisonRequest),
    ],
)
def test_broker_failure_is_recorded_without_running_engine(
    run_type, request_type, monkeypatch
):
    def unavailable(*args):
        raise RuntimeError("Broker unavailable")

    def must_not_build(*args):
        pytest.fail("The HTTP submission path must not prepare engine parameters")

    monkeypatch.setattr("api.worker.run_coordinator_task.delay", unavailable)
    monkeypatch.setattr("api.run_execution.prepare_run", must_not_build)
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = UserRow(email="submission@example.com", password_hash="unused")
        db.add(user)
        db.commit()
        store = DBRunStore()
        with pytest.raises(RuntimeError, match="Broker unavailable"):
            submit_run(request_type(), run_type, store, user.id, db)
        saved = store.list_all(db=db, user_id=user.id)
        assert len(saved) == 1
        assert saved[0].status == "failed"
        assert "Could not enqueue run" in saved[0].error_message
    engine.dispose()
