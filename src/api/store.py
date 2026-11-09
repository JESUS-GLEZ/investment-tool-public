"""In-memory run store.

A simple thread-safe dict-based store for run records.  Designed to be
swapped for a database-backed implementation in Phase 1b without
changing the router code.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any
from collections.abc import Callable

from api.schemas import RunDetail, RunStatus, RunType


class RunStore:
    """Thread-safe in-memory store for run records.

    Each run is stored as a :class:`RunDetail`.  In Phase 1b this will
    be replaced by a SQLAlchemy/DB-backed implementation with the same
    public interface.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunDetail] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def create(
        self,
        run_type: RunType,
        request_payload: dict[str, Any],
    ) -> RunDetail:
        """Create a new run in ``QUEUED`` status and return it."""
        run_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc)
        run = RunDetail(
            id=run_id,
            run_type=run_type,
            status=RunStatus.QUEUED,
            created_at=now,
            request_payload=request_payload,
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def mark_running(self, run_id: str) -> None:
        """Transition a run to ``RUNNING``."""
        with self._lock:
            run = self._runs[run_id]
            self._runs[run_id] = run.model_copy(
                update={"status": RunStatus.RUNNING},
            )

    def mark_completed(
        self,
        run_id: str,
        result: dict[str, Any],
    ) -> None:
        """Transition a run to ``COMPLETED`` and attach the result."""
        with self._lock:
            run = self._runs[run_id]
            self._runs[run_id] = run.model_copy(
                update={
                    "status": RunStatus.COMPLETED,
                    "completed_at": datetime.now(timezone.utc),
                    "result": result,
                },
            )

    def mark_failed(
        self,
        run_id: str,
        error_message: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Transition a run to ``FAILED`` and record the error."""
        with self._lock:
            run = self._runs[run_id]
            self._runs[run_id] = run.model_copy(
                update={
                    "status": RunStatus.FAILED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": error_message,
                    "result": result if result is not None else run.result,
                },
            )

    def mark_cancelled(
        self,
        run_id: str,
        result: dict[str, Any] | None = None,
    ) -> None:
        """Transition a run to ``CANCELLED``, optionally attaching partial results."""
        with self._lock:
            run = self._runs[run_id]
            self._runs[run_id] = run.model_copy(
                update={
                    "status": RunStatus.CANCELLED,
                    "completed_at": datetime.now(timezone.utc),
                    "error_message": "Cancelled by user",
                    "result": result if result is not None else run.result,
                },
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, run_id: str) -> RunDetail | None:
        """Return a run by ID, or ``None`` if not found."""
        with self._lock:
            return self._runs.get(run_id)

    def list_all(self) -> list[RunDetail]:
        """Return all runs ordered by creation time (newest first)."""
        with self._lock:
            runs: list[RunDetail] = sorted(
                self._runs.values(),
                key=lambda r: r.created_at,
                reverse=True,
            )
            return runs

    def clear(self) -> None:
        """Remove all runs (useful for tests)."""
        with self._lock:
            self._runs.clear()

    def delete(self, run_id: str, *, cleanup: Callable[[], None]) -> None:
        with self._lock:
            run = self._runs[run_id]
            if run.status not in {"completed", "failed", "cancelled"}:
                raise ValueError("Stop the run before deleting it")
            cleanup()
            del self._runs[run_id]
