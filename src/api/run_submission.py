"""Shared run submission; HTTP routes do not execute models or build results."""

from __future__ import annotations

import logging
import threading
from typing import TypeAlias

from sqlalchemy.orm import Session

from api.campaigns import validate_campaign
from api.db_store import DBRunStore
from api.schemas import (
    ExecutionRequest,
    PredictionComparisonRequest,
    RunSummary,
    RunType,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)
from api.store import RunStore

logger = logging.getLogger(__name__)
RunRequest: TypeAlias = (
    ExecutionRequest
    | StrategyComparisonRequest
    | SolverComparisonRequest
    | PredictionComparisonRequest
)


def submit_run(
    body: RunRequest,
    run_type: RunType,
    store: RunStore | DBRunStore,
    user_id: int,
    db: Session,
) -> RunSummary:
    validate_campaign(store, db, body.campaign_id, user_id)
    payload = body.model_dump(mode="json", exclude={"campaign_id"})
    if isinstance(store, DBRunStore):
        run = store.create(
            run_type=run_type,
            request_payload=payload,
            db=db,
            user_id=user_id,
            campaign_id=body.campaign_id,
        )
        from api.worker import run_coordinator_task

        try:
            task = run_coordinator_task.delay(run.id, run_type.value, payload)
        except Exception as exc:
            store.mark_failed(run.id, f"Could not enqueue run: {exc}", db=db)
            raise
        store.set_celery_task_id(run.id, task.id, db=db)
        return RunSummary(
            id=run.id,
            run_type=run.run_type,
            status=run.status,
            created_at=run.created_at,
            completed_at=run.completed_at,
            error_message=run.error_message,
            campaign_id=run.campaign_id,
        )

    from api.run_execution import prepare_run
    from api.result_presenter import present_result
    from api.sanitize import sanitize_output

    run = store.create(run_type=run_type, request_payload=payload)
    try:
        prepared = prepare_run(run_type.value, payload, run.id)
    except Exception as exc:
        store.mark_failed(run.id, str(exc))
        raise

    def work() -> None:
        try:
            result = sanitize_output(
                present_result(prepared.execute(), run_type.value),
                run_type.value,
                body.output_mode,
            )
            store.mark_completed(run.id, result)
        except Exception as exc:
            logger.exception("Run %s failed", run.id)
            store.mark_failed(run.id, str(exc))

    store.mark_running(run.id)
    threading.Thread(target=work, name=f"run-{run.id}", daemon=True).start()
    detail = store.get(run.id)
    assert detail is not None
    return RunSummary(**detail.model_dump())
