"""Comparisons router — multi-strategy / solver / prediction endpoints.

Endpoints
---------
POST /comparisons/strategy     Enqueue a strategy comparison (async via Celery).
POST /comparisons/solver       Enqueue a solver comparison.
POST /comparisons/prediction   Enqueue a prediction comparison.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import get_db
from api.db_store import DBRunStore
from api.models import UserRow
from api.schemas import (
    PredictionComparisonRequest,
    RunSummary,
    RunType,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)
from api.run_submission import submit_run
from api.store import RunStore

router = APIRouter(prefix="/comparisons", tags=["comparisons"])


# ---------------------------------------------------------------------------
# Store dependency (same pattern as runs router)
# ---------------------------------------------------------------------------

_store: RunStore | DBRunStore | None = None


def set_store(store: RunStore | DBRunStore) -> None:
    """Called once at app startup to wire the store singleton."""
    global _store  # noqa: PLW0603
    _store = store


def _get_store() -> RunStore | DBRunStore:
    assert _store is not None, "RunStore not initialised — call set_store() first"
    return _store


Store = Annotated[RunStore | DBRunStore, Depends(_get_store)]
CurrentUser = Annotated[UserRow, Depends(get_current_user)]
DB = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/strategy", response_model=RunSummary, status_code=202)
def create_strategy_comparison(
    body: StrategyComparisonRequest,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> RunSummary:
    """Submit a strategy comparison run."""
    return submit_run(body, RunType.STRATEGY_COMPARISON, store, user.id, db)


@router.post("/solver", response_model=RunSummary, status_code=202)
def create_solver_comparison(
    body: SolverComparisonRequest,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> RunSummary:
    """Submit a solver comparison run."""
    return submit_run(body, RunType.SOLVER_COMPARISON, store, user.id, db)


@router.post("/prediction", response_model=RunSummary, status_code=202)
def create_prediction_comparison(
    body: PredictionComparisonRequest,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> RunSummary:
    """Submit a prediction comparison run."""
    return submit_run(body, RunType.PREDICTION_COMPARISON, store, user.id, db)
