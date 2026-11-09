"""Runs router — single-strategy execution endpoints.

Endpoints
---------
POST /runs                              Enqueue a single-strategy run (async via Celery).
GET  /runs                              List all runs for the authenticated user.
GET  /runs/{id}                         Get a single run by ID.
DELETE /runs/{id}                      Delete a terminal run and its own outputs.
GET  /runs/{id}/files                   List output files produced by a run.
GET  /runs/{id}/files/download-all      Download all output files as a ZIP archive.
GET  /runs/{id}/files/{file_path}       Download a specific output file.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from api.auth import get_current_user
from api.database import get_db
from api.campaigns import validate_campaign
from api.db_store import DBRunStore
from api.models import UserRow
from api.schemas import (
    ExecutionRequest,
    RunDetail,
    RunSummary,
    RunType,
    RunCampaignRequest,
)
from api.adapters import _OUTPUT_BASE
from api.cancel import request_cancel
from api.run_submission import submit_run
from api.store import RunStore
from api.run_cleanup import delete_run_outputs, ensure_cancelled_task_stopped

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


# ---------------------------------------------------------------------------
# Dependency — the RunStore is injected via FastAPI's dependency system.
# The actual instance is wired in ``api.main.create_app()``.
# ---------------------------------------------------------------------------

_store: RunStore | DBRunStore | None = None


def set_store(store: RunStore | DBRunStore) -> None:
    """Called once at app startup to wire the store singleton."""
    global _store  # noqa: PLW0603
    _store = store


def _get_store() -> RunStore | DBRunStore:
    """FastAPI dependency that returns the run store."""
    assert _store is not None, "RunStore not initialised — call set_store() first"
    return _store


Store = Annotated[RunStore | DBRunStore, Depends(_get_store)]
CurrentUser = Annotated[UserRow, Depends(get_current_user)]
DB = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("", response_model=RunSummary, status_code=202)
def create_run(
    body: ExecutionRequest,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> RunSummary:
    """Submit a single-strategy execution run.

    The run is enqueued for async execution via Celery.  Returns
    **202 Accepted** with a :class:`RunSummary`.
    """
    return submit_run(body, RunType.EXECUTION, store, user.id, db)


@router.get("", response_model=list[RunSummary])
def list_runs(store: Store, user: CurrentUser, db: DB) -> list[RunSummary]:
    """Return all runs for the authenticated user (newest first)."""
    if isinstance(store, DBRunStore):
        return store.list_all(db=db, user_id=user.id)  # type: ignore[no-any-return]
    # In-memory fallback — RunStore.list_all() returns list[RunDetail] which
    # is a superset of RunSummary; FastAPI's response_model handles trimming.
    return store.list_all()  # type: ignore[no-any-return]


@router.put("/{run_id}/campaign", response_model=RunSummary)
def assign_campaign(
    run_id: str, body: RunCampaignRequest, store: Store, user: CurrentUser, db: DB
) -> RunDetail:
    """Change grouping metadata only; never modify or enqueue a run."""
    if not isinstance(store, DBRunStore):
        raise HTTPException(501, "Campaigns require database-backed run storage")
    from api.models import RunRow

    row = (
        db.query(RunRow)
        .filter_by(id=run_id, user_id=user.id)
        .with_for_update(of=RunRow)
        .first()
    )
    if row is None:
        raise HTTPException(404, "Run not found")
    validate_campaign(store, db, body.campaign_id, user.id)
    row.campaign_id = body.campaign_id
    db.commit()
    detail = store.get_for_user(run_id, user.id, db=db)
    assert detail is not None
    return detail


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: str, store: Store, user: CurrentUser, db: DB) -> RunDetail:
    """Return a single run by ID, or 404."""
    if isinstance(store, DBRunStore):
        run = store.get_for_user(run_id, user.id, db=db)
    else:
        run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{run_id}/cancel", response_model=RunSummary)
def cancel_run(run_id: str, store: Store, user: CurrentUser, db: DB) -> RunSummary:
    """Cancel a run and interrupt its Celery task without awaiting solver progress.

    Already-cancelled requests are retryable. Previously saved output is kept;
    unsaved in-memory solver work may be lost on process termination.
    """
    if isinstance(store, DBRunStore):
        run = store.get_for_user(run_id, user.id, db=db)
    else:
        run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in ("queued", "running", "cancelled"):
        raise HTTPException(
            status_code=409,
            detail=f"Run is already {run.status} and cannot be cancelled",
        )

    if not isinstance(store, DBRunStore) and run.status == "running":
        raise HTTPException(
            501, "Stopping an active run requires the Celery worker deployment."
        )

    try:
        request_cancel(run_id)
        if isinstance(store, DBRunStore):
            from api.models import RunRow
            from api.worker import celery_app

            row = db.get(RunRow, run_id)
            if row is None:
                raise HTTPException(status_code=404, detail="Run not found")
            if row.celery_task_id:
                # Interrupt the prefork child even while it is inside a solver.
                celery_app.control.revoke(
                    row.celery_task_id, terminate=True, signal="SIGTERM"
                )
            store.mark_cancelled(run_id, db=db)
        elif run.status == "queued":
            store.mark_cancelled(run_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Cancellation dispatch failed for %s", run_id)
        raise HTTPException(
            503, "Could not confirm cancellation dispatch. Please retry."
        ) from exc

    # Re-fetch and return current state
    if isinstance(store, DBRunStore):
        updated = store.get_for_user(run_id, user.id, db=db)
    else:
        updated = store.get(run_id)
    return RunSummary(
        id=updated.id,  # type: ignore[union-attr]
        run_type=updated.run_type,  # type: ignore[union-attr]
        status=updated.status,  # type: ignore[union-attr]
        created_at=updated.created_at,  # type: ignore[union-attr]
        completed_at=updated.completed_at,  # type: ignore[union-attr]
        error_message=updated.error_message,  # type: ignore[union-attr]
    )


def _run_output_dir(run_id: str) -> Path:
    base = Path(_OUTPUT_BASE).resolve()
    directory = base / run_id
    # Match cleanup's boundary: a run root must be a real direct child of
    # the configured output base, rather than a link to another run or folder.
    if directory.is_symlink() or directory.resolve().parent != base:
        raise HTTPException(status_code=400, detail="Invalid run output directory")
    return directory


def _is_output_file(path: Path, directory: Path) -> bool:
    """Only export regular files through real directories inside this run.

    Output trees are operator-controlled. This check excludes existing links;
    it does not isolate the API from concurrent hostile filesystem mutation.
    """
    try:
        relative = path.relative_to(directory)
        current = directory
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                return False
        return path.resolve().is_relative_to(directory) and path.is_file()
    except (OSError, RuntimeError, ValueError):
        return False


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: str, store: Store, user: CurrentUser, db: DB) -> Response:
    """Permanently delete an owned terminal run and its output directory.

    Shared input snapshots, market/news caches and presets remain available.
    Active runs must be stopped first. A failed filesystem cleanup retains the
    database record so the operation can be retried.
    """
    run = (
        store.get_for_user(run_id, user.id, db=db)
        if isinstance(store, DBRunStore)
        else store.get(run_id)
    )
    if run is None:
        raise HTTPException(404, "Run not found")

    def cleanup() -> None:
        if isinstance(store, DBRunStore) and run.status == "cancelled":
            from api.models import RunRow

            row = db.get(RunRow, run_id)
            if row is not None and row.celery_task_id:
                ensure_cancelled_task_stopped(row.celery_task_id)
        delete_run_outputs(Path(_OUTPUT_BASE), run_id)

    try:
        if isinstance(store, DBRunStore):
            store.delete_for_user(run_id, user.id, db=db, cleanup=cleanup)
        else:
            store.delete(run_id, cleanup=cleanup)
    except KeyError as exc:
        raise HTTPException(404, "Run not found") from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        logger.exception("Run deletion failed for %s", run_id)
        raise HTTPException(
            503, "Run deletion could not finish. Please retry."
        ) from exc
    return Response(status_code=204)


@router.get("/{run_id}/files", response_model=list[dict])
def list_run_files(run_id: str, store: Store, user: CurrentUser, db: DB) -> list[dict]:
    """List all output files produced by a completed run.

    Returns a list of ``{"path": str, "size": int, "type": str}`` objects,
    where ``path`` is relative to the run's output directory.
    """
    if isinstance(store, DBRunStore):
        run = store.get_for_user(run_id, user.id, db=db)
    else:
        run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    out_dir = _run_output_dir(run_id)
    if not out_dir.exists():
        return []

    files = []
    for path in sorted(out_dir.rglob("*")):
        if path.name == ".keep" or not _is_output_file(path, out_dir):
            continue
        rel = path.relative_to(out_dir)
        mime, _ = mimetypes.guess_type(path.name)
        files.append(
            {
                "path": str(rel),
                "size": path.stat().st_size,
                "type": mime or "application/octet-stream",
            }
        )
    return files


@router.get("/{run_id}/files/download-all")
def download_all_run_files(
    run_id: str,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> StreamingResponse:
    """Stream all output files for a run as a single ZIP archive.

    The archive preserves the directory structure inside the run's output
    directory.  The response is streamed in-memory so no temporary file is
    written to disk.
    """
    if isinstance(store, DBRunStore):
        run = store.get_for_user(run_id, user.id, db=db)
    else:
        run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    out_dir = _run_output_dir(run_id)
    if not out_dir.exists():
        raise HTTPException(
            status_code=404, detail="No output files found for this run"
        )

    files = [
        p
        for p in sorted(out_dir.rglob("*"))
        if p.name != ".keep" and _is_output_file(p, out_dir)
    ]
    if not files:
        raise HTTPException(
            status_code=404, detail="No output files found for this run"
        )

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            arcname = str(path.relative_to(out_dir))
            zf.write(path, arcname=arcname)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="run_{run_id}_files.zip"'
        },
    )


@router.get("/{run_id}/files/{file_path:path}")
def download_run_file(
    run_id: str,
    file_path: str,
    store: Store,
    user: CurrentUser,
    db: DB,
) -> FileResponse:
    """Download a specific output file from a run.

    ``file_path`` is relative to the run's output directory, e.g.
    ``plots/equity_curve.png`` or ``backtest_results.csv``.
    """
    if isinstance(store, DBRunStore):
        run = store.get_for_user(run_id, user.id, db=db)
    else:
        run = store.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    out_dir = _run_output_dir(run_id)
    target = out_dir / file_path

    # Guard against path traversal
    try:
        target.resolve().relative_to(out_dir)
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid file path")

    # Check the original path too: resolving first loses symlink components.
    if target.is_symlink() or (
        target.is_file() and not _is_output_file(target, out_dir)
    ):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    mime, _ = mimetypes.guess_type(target.name)
    return FileResponse(
        path=target,
        filename=target.name,
        media_type=mime or "application/octet-stream",
    )
