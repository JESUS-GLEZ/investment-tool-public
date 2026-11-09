"""Celery application and background tasks.

The Celery app is configured with Redis as both broker and result
backend.  The main task :func:`run_coordinator_task` executes a
coordinator function and updates the DB store with the result.

Progress is streamed to Redis pub/sub via :mod:`api.progress` so that
WebSocket clients can display real-time updates.
"""

from __future__ import annotations

import logging
from typing import Any

from celery import Celery
from celery.signals import task_failure, task_revoked

from api.result_presenter import present_result
from investment_tool.config import get_config
from investment_tool.exceptions import BacktestReadinessError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Celery app
# ---------------------------------------------------------------------------


def make_celery_app() -> Celery:
    """Build and return a configured :class:`Celery` application."""
    redis_url = get_config().redis.url
    app = Celery(
        "investment_tool_worker",
        broker=redis_url,
        backend=redis_url,
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        # No hard time limit — some jobs (large EA runs) can take days.
        # Set to 7 days to act only as a stuck-task safety net.
        task_time_limit=604800,
        task_soft_time_limit=604200,
    )
    return app


celery_app = make_celery_app()


# ---------------------------------------------------------------------------
# Coordinator dispatch
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Result enrichment
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Celery task
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="run_coordinator")  # type: ignore[misc]
def run_coordinator_task(
    self: Any,
    run_id: str,
    run_type: str,
    request_payload: dict[str, Any],
) -> dict[str, Any]:
    """Execute a coordinator and persist the result to the DB.

    This task is enqueued by the API endpoints.  It:

    1. Marks the run as RUNNING.
    2. Creates a Redis-backed progress callback.
    3. Builds coordinator parameters from the request payload.
    4. Calls the coordinator function with the progress callback.
    5. Sanitizes the result and marks the run COMPLETED (or FAILED).
    6. Publishes a sentinel message on completion/failure.
    """
    from api.run_execution import prepare_run
    from api.cancel import RunCancelledError, clear_cancel, is_cancel_requested
    from api.db_store import DBRunStore
    from api.database import get_session_factory
    from api.progress import make_progress_publisher, publish_sentinel
    from api.sanitize import sanitize_output
    from investment_tool.output.policy import OutputMode
    from api.run_monitor import RunMonitor

    store = DBRunStore()
    session_factory = get_session_factory()
    db = session_factory()
    monitor = RunMonitor(run_id, session_factory)
    partial_sink: dict[str, Any] = {}
    output_mode = OutputMode.STANDARD

    # Set up Redis progress publisher
    redis_client = None
    progress_callback = None
    try:
        redis_client, progress_callback = make_progress_publisher(run_id)
    except Exception:
        logger.warning(
            "Could not create progress publisher for run %s; "
            "progress will not be streamed",
            run_id,
            exc_info=True,
        )

    # Dedicated Redis client for cancellation checks (always created so
    # cancel detection works even if the progress publisher failed).
    from investment_tool.config import get_config as _get_config
    import redis as _redis_mod

    _cancel_client: _redis_mod.Redis | None = None  # type: ignore[type-arg]
    try:
        _cancel_client = _redis_mod.from_url(
            _get_config().redis.url, decode_responses=True
        )
    except Exception:
        logger.warning(
            "Could not create Redis client for cancellation checks for run %s", run_id
        )

    try:
        if _cancel_client is not None and is_cancel_requested(run_id, _cancel_client):
            store.mark_cancelled(run_id, db=db, result=None)
            clear_cancel(run_id, _cancel_client)
            publish_sentinel(run_id, status=store.get(run_id, db=db).status.value)
            return {"status": "cancelled", "run_id": run_id}

        # 1. Mark running
        if store.mark_running(run_id, db=db) is False:
            return {"status": "skipped", "run_id": run_id}
        monitor.start()

        # 2. Build params and call the coordinator
        prepared = prepare_run(run_type, request_payload, run_id)
        params = prepared.parameters

        output_mode = params.output_mode

        # Wrap progress_callback to cooperatively check cancellation flag.
        # A partial_sink dict is passed to coordinators that support it so
        # we can recover per-iteration results even if the run is cancelled
        # mid-loop.

        _inner_cb = progress_callback

        def _cancellable_progress(frac: float, msg: str) -> None:
            monitor.progress(frac, msg)
            if _cancel_client is not None and is_cancel_requested(
                run_id, _cancel_client
            ):
                raise RunCancelledError(f"Run {run_id} cancelled by user")
            if _inner_cb is not None:
                _inner_cb(frac, msg)

        logger.info("Running coordinator for run %s (type=%s)", run_id, run_type)

        def checkpoint_partial(partial: dict[str, Any]) -> None:
            try:
                store.save_partial(
                    run_id, sanitize_output(partial, run_type, output_mode), db=db
                )
            except Exception:
                db.rollback()
                logger.warning("Could not checkpoint run %s", run_id, exc_info=True)

        raw_result = prepared.execute(
            _cancellable_progress, partial_sink, checkpoint_partial
        )

        monitor.progress(monitor.fraction, "Preparing and saving results")

        # Clear any stale cancel flag now that the run finished normally.
        try:
            if _cancel_client is not None:
                clear_cancel(run_id, _cancel_client)
        except Exception:
            pass

        # 3. Enrich result with Plotly figures and normalised comparison metrics
        raw_result = present_result(raw_result, run_type)

        # 4. Sanitize and persist
        result = sanitize_output(raw_result, run_type, output_mode)
        store.mark_completed(run_id, result, db=db)
        logger.info("Run %s completed successfully", run_id)

        # 5. Publish completion sentinel
        try:
            final_status = store.get(run_id, db=db).status.value
            publish_sentinel(run_id, status=final_status)
        except Exception:
            logger.debug("Failed to publish sentinel for %s", run_id, exc_info=True)

        return {"status": store.get(run_id, db=db).status.value, "run_id": run_id}

    except RunCancelledError:  # type: ignore[misc]
        logger.info("Run %s was cancelled; persisting partial results", run_id)
        partial_result: dict[str, Any] | None = None
        if partial_sink:
            try:
                partial_result = present_result(dict(partial_sink), run_type)
                partial_result = sanitize_output(partial_result, run_type, output_mode)
            except Exception:
                logger.warning(
                    "Failed to enrich/sanitize partial result for run %s",
                    run_id,
                    exc_info=True,
                )
                partial_result = None
        try:
            store.mark_cancelled(run_id, db=db, result=partial_result)
        except Exception:
            logger.exception("Failed to mark run %s as cancelled in DB", run_id)
        try:
            if _cancel_client is not None:
                clear_cancel(run_id, _cancel_client)
        except Exception:
            pass
        try:
            publish_sentinel(run_id, status=store.get(run_id, db=db).status.value)
        except Exception:
            pass
        return {"status": "cancelled", "run_id": run_id}

    except Exception as exc:
        logger.exception("Run %s failed", run_id)
        if isinstance(exc, BacktestReadinessError):
            partial_sink["validation_report"] = exc.validation_report
        partial_result = None
        if partial_sink:
            try:
                partial_result = sanitize_output(
                    present_result(dict(partial_sink), run_type), run_type, output_mode
                )
            except Exception:
                logger.warning(
                    "Could not prepare partial result for run %s", run_id, exc_info=True
                )
        try:
            store.mark_failed(run_id, str(exc), db=db, result=partial_result)
        except Exception:
            logger.exception("Failed to mark run %s as failed in DB", run_id)

        # Publish failure sentinel
        try:
            publish_sentinel(
                run_id, status=store.get(run_id, db=db).status.value, error=str(exc)
            )
        except Exception:
            logger.debug(
                "Failed to publish failure sentinel for %s", run_id, exc_info=True
            )

        return {
            "status": store.get(run_id, db=db).status.value,
            "run_id": run_id,
            "error": str(exc),
        }

    except BaseException as exc:  # catches SoftTimeLimitExceeded etc.
        msg = f"Task interrupted: {type(exc).__name__}: {exc}"
        logger.error("Run %s interrupted by BaseException: %s", run_id, msg)
        try:
            partial_result = (
                sanitize_output(dict(partial_sink), run_type, output_mode)
                if partial_sink
                else None
            )
            store.mark_failed(run_id, msg, db=db, result=partial_result)
        except Exception:
            logger.exception("Failed to mark run %s as failed in DB", run_id)
        try:
            publish_sentinel(
                run_id, status=store.get(run_id, db=db).status.value, error=msg
            )
        except Exception:
            pass
        raise  # re-raise so Celery records it correctly

    finally:
        if monitor.thread.ident is not None:
            monitor.stop()
        if redis_client is not None:
            try:
                redis_client.close()
            except Exception:
                pass
        if _cancel_client is not None:
            try:
                _cancel_client.close()
            except Exception:
                pass
        db.close()


# These run in the parent worker as well, covering abrupt pool-process loss.


def record_worker_failure(
    sender: Any = None,
    task_id: str | None = None,
    exception: BaseException | None = None,
    **kwargs: Any,
) -> None:
    _record_terminal_task(task_id, "failed", f"Worker task failed: {exception}")


def record_worker_revocation(request: Any = None, **kwargs: Any) -> None:
    _record_terminal_task(getattr(request, "id", None), "cancelled", "Task revoked")


def _record_terminal_task(task_id: str | None, status: str, message: str) -> None:
    if not task_id:
        return
    from api.database import get_session_factory
    from api.db_store import DBRunStore
    from api.models import RunRow
    from api.progress import publish_sentinel

    try:
        with get_session_factory()() as db:
            row = db.query(RunRow).filter_by(celery_task_id=task_id).first()
            if row is None:
                return
            run_id = row.id
            store = DBRunStore()
            if status == "cancelled":
                store.mark_cancelled(run_id, db=db)
            else:
                store.mark_failed(run_id, message, db=db)
            db.refresh(row)
            publish_sentinel(run_id, status=row.status)
    except Exception:
        logger.exception("Could not record termination of task %s", task_id)


task_failure.connect(record_worker_failure)
task_revoked.connect(record_worker_revocation)
