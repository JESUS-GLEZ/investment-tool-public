"""Database-backed run store.

Provides the same public interface as :class:`api.store.RunStore` but
persists data via SQLAlchemy sessions.  The router code stays unchanged
because both stores expose identical method signatures.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from sqlalchemy.orm import Session, lazyload

from api.models import RunResultRow, RunRow
from api.schemas import RunDetail, RunStatus, RunSummary, RunType


class DBRunStore:
    """SQLAlchemy-backed run store.

    Unlike the in-memory :class:`RunStore` (which is an object you keep
    a reference to), this class requires a :class:`Session` per call so
    that it plays nicely with FastAPI's ``Depends(get_db)`` lifecycle.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create(
        self,
        run_type: RunType,
        request_payload: dict[str, Any],
        *,
        db: Session,
        user_id: int,
        campaign_id: str | None = None,
    ) -> RunDetail:
        """Create a new run in ``QUEUED`` status and return it."""
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        row = RunRow(
            id=run_id,
            user_id=user_id,
            campaign_id=campaign_id,
            run_type=run_type.value,
            status=RunStatus.QUEUED.value,
            request_payload=request_payload,
            created_at=now,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return _row_to_detail(row)

    def mark_running(self, run_id: str, *, db: Session) -> bool:
        """Transition a run to ``RUNNING``."""
        row = _get_row(db, run_id)
        if row.status != RunStatus.QUEUED.value:
            db.commit()
            return False
        row.status = RunStatus.RUNNING.value
        row.heartbeat_at = datetime.now(timezone.utc)
        row.progress_message = "Preparing run"
        db.commit()
        return True

    def mark_completed(
        self,
        run_id: str,
        result: dict[str, Any],
        *,
        db: Session,
    ) -> None:
        """Transition a run to ``COMPLETED`` and attach the result."""
        row = _get_row(db, run_id)
        if row.status not in (RunStatus.QUEUED.value, RunStatus.RUNNING.value):
            db.commit()
            return
        row.status = RunStatus.COMPLETED.value
        row.completed_at = datetime.now(timezone.utc)

        # Upsert the result row
        if row.result is not None:
            row.result.result = result
        else:
            result_row = RunResultRow(run_id=run_id, result=result)
            db.add(result_row)

        db.commit()

    def save_partial(self, run_id: str, result: dict[str, Any], *, db: Session) -> None:
        """Checkpoint completed work while a run is still active."""
        row = _get_row(db, run_id)
        if row.status != RunStatus.RUNNING.value:
            db.commit()
            return
        if row.result is not None:
            row.result.result = result
        else:
            db.add(RunResultRow(run_id=run_id, result=result))
        db.commit()

    def mark_failed(
        self,
        run_id: str,
        error_message: str,
        *,
        db: Session,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Transition a run to ``FAILED`` and record the error."""
        row = _get_row(db, run_id)
        if row.status not in (RunStatus.QUEUED.value, RunStatus.RUNNING.value):
            db.commit()
            return
        row.status = RunStatus.FAILED.value
        row.completed_at = datetime.now(timezone.utc)
        row.error_message = error_message
        if result is not None:
            if row.result is not None:
                row.result.result = result
            else:
                db.add(RunResultRow(run_id=run_id, result=result))
        db.commit()

    def mark_cancelled(
        self,
        run_id: str,
        *,
        db: Session,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Transition a run to ``CANCELLED``, optionally attaching partial results."""
        row = _get_row(db, run_id)
        if row.status not in (RunStatus.QUEUED.value, RunStatus.RUNNING.value):
            db.commit()
            return
        row.status = RunStatus.CANCELLED.value
        row.completed_at = datetime.now(timezone.utc)
        row.error_message = "Cancelled by user"
        if result is not None:
            if row.result is not None:
                row.result.result = result
            else:
                result_row = RunResultRow(run_id=run_id, result=result)
                db.add(result_row)
        db.commit()

    def set_celery_task_id(self, run_id: str, task_id: str, *, db: Session) -> None:
        """Record the Celery task ID on a run."""
        row = _get_row(db, run_id)
        row.celery_task_id = task_id
        db.commit()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: str, *, db: Session) -> RunDetail | None:
        """Return a run by ID, or ``None`` if not found."""
        row = db.query(RunRow).filter(RunRow.id == run_id).first()
        if row is None:
            return None
        return _row_to_detail(row)

    def get_for_user(
        self, run_id: str, user_id: int, *, db: Session
    ) -> RunDetail | None:
        """Return a run by ID only if it belongs to *user_id*.

        Returns ``None`` if the run does not exist **or** if it belongs
        to a different user.  Callers should raise HTTP 404 in both cases
        to avoid leaking run-existence information.
        """
        row = (
            db.query(RunRow)
            .filter(RunRow.id == run_id, RunRow.user_id == user_id)
            .first()
        )
        if row is None:
            return None
        return _row_to_detail(row)

    def list_all(
        self,
        *,
        db: Session,
        user_id: int | None = None,
    ) -> list[RunSummary]:
        """Return runs ordered by creation time (newest first).

        If *user_id* is given, only that user's runs are returned.
        """
        query = (
            db.query(RunRow)
            .options(lazyload(RunRow.result))
            .order_by(RunRow.created_at.desc())
        )
        if user_id is not None:
            query = query.filter(RunRow.user_id == user_id)
        rows = query.all()
        return [_row_to_summary(r) for r in rows]

    def clear(self, *, db: Session) -> None:
        """Remove all runs (useful for tests)."""
        db.query(RunResultRow).delete()
        db.query(RunRow).delete()
        db.commit()

    def delete_for_user(
        self,
        run_id: str,
        user_id: int,
        *,
        db: Session,
        cleanup: Callable[[], None],
    ) -> None:
        """Delete a terminal run, cleaning files before committing metadata removal."""
        row = (
            db.query(RunRow)
            .filter(RunRow.id == run_id, RunRow.user_id == user_id)
            .populate_existing()
            .with_for_update(of=RunRow)
            .first()
        )
        try:
            if row is None:
                raise KeyError(run_id)
            if row.status not in {"completed", "failed", "cancelled"}:
                raise ValueError("Stop the run before deleting it")
            cleanup()
            db.delete(row)
            db.commit()
        except Exception:
            db.rollback()
            raise


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_row(db: Session, run_id: str) -> RunRow:
    """Fetch a run row or raise ``KeyError`` (matching in-memory store)."""
    row = (
        db.query(RunRow)
        .filter(RunRow.id == run_id)
        .populate_existing()
        .with_for_update(of=RunRow)
        .first()
    )
    if row is None:
        raise KeyError(f"Run {run_id!r} not found")
    return row


def _row_to_detail(row: RunRow) -> RunDetail:
    """Convert a :class:`RunRow` (+ optional result) to :class:`RunDetail`."""
    result_data = row.result.result if row.result is not None else None
    return RunDetail(
        id=row.id,
        run_type=RunType(row.run_type),
        status=RunStatus(row.status),
        created_at=row.created_at,
        completed_at=row.completed_at,
        error_message=row.error_message,
        campaign_id=row.campaign_id,
        heartbeat_at=row.heartbeat_at,
        worker_health=_worker_health(row),
        progress_fraction=row.progress_fraction,
        progress_message=row.progress_message,
        request_payload=row.request_payload,
        result=result_data,
    )


def _row_to_summary(row: RunRow) -> RunSummary:
    """Convert a :class:`RunRow` to :class:`RunSummary` (no result payload)."""
    return RunSummary(
        id=row.id,
        run_type=RunType(row.run_type),
        status=RunStatus(row.status),
        created_at=row.created_at,
        completed_at=row.completed_at,
        error_message=row.error_message,
        campaign_id=row.campaign_id,
        heartbeat_at=row.heartbeat_at,
        worker_health=_worker_health(row),
        progress_fraction=row.progress_fraction,
        progress_message=row.progress_message,
        request_payload=row.request_payload,
    )


HEARTBEAT_TIMEOUT_SECONDS = 120


def _worker_health(row: RunRow) -> str | None:
    if row.status != RunStatus.RUNNING.value:
        return None
    if row.heartbeat_at is None:
        return "unknown"
    heartbeat = (
        row.heartbeat_at.replace(tzinfo=timezone.utc)
        if row.heartbeat_at.tzinfo is None
        else row.heartbeat_at
    )
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    return "unresponsive" if age > HEARTBEAT_TIMEOUT_SECONDS else "alive"
