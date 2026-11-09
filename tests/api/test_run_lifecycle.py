"""Liveness, cancellation dispatch and terminal-state races."""

from datetime import datetime, timedelta, timezone
from threading import Event
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.auth import get_current_user
from api.database import get_db
from api.db_store import DBRunStore
from api.models import Base, RunRow, UserRow
from api.routers import runs
from api.run_monitor import RunMonitor
from api.schemas import RunType


@pytest.fixture()
def context(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'runs.db'}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as db:
        user = UserRow(email="owner@example.com", password_hash="unused")
        db.add(user)
        db.commit()
        store = DBRunStore()
        run = store.create(RunType.SOLVER_COMPARISON, {}, db=db, user_id=user.id)
        store.set_celery_task_id(run.id, "task-123", db=db)
        yield factory, db, store, run, user
    engine.dispose()


def test_heartbeat_runs_without_solver_callbacks(context):
    factory, db, store, run, _ = context
    store.mark_running(run.id, db=db)
    monitor = RunMonitor(run.id, factory, interval=0.01)
    monitor.progress(0.25, "NSGA-II: restart 1/3")
    observed = Event()
    original = monitor.beat

    def beat():
        result = original()
        observed.set()
        return result

    monitor.beat = beat
    monitor.start()
    try:
        assert observed.wait(2)
    finally:
        monitor.stop()
    db.expire_all()
    state = store.get(run.id, db=db)
    assert state.worker_health == "alive"
    assert state.progress_fraction == 0.25
    assert "NSGA-II" in state.progress_message
    assert not monitor.thread.is_alive()


def test_stale_heartbeat_is_not_misreported_as_alive_or_failed(context):
    _, db, store, run, _ = context
    store.mark_running(run.id, db=db)
    row = db.get(RunRow, run.id)
    row.heartbeat_at = datetime.now(timezone.utc) - timedelta(minutes=3)
    db.commit()
    assert store.get(run.id, db=db).worker_health == "unresponsive"
    assert store.get(run.id, db=db).status == "running"
    row.heartbeat_at = datetime.now(timezone.utc)
    db.commit()
    assert store.get(run.id, db=db).worker_health == "alive"


def test_cancellation_cannot_be_overwritten_by_late_result(context):
    factory, db, store, run, _ = context
    store.mark_running(run.id, db=db)
    with factory() as worker_db:
        worker_db.get(RunRow, run.id)  # Simulate a worker's stale identity map.
        store.mark_cancelled(run.id, db=db)
        store.mark_completed(run.id, {"late": True}, db=worker_db)
        store.mark_failed(run.id, "late failure", db=worker_db)
        assert store.mark_running(run.id, db=worker_db) is False
    db.expire_all()
    state = store.get(run.id, db=db)
    assert state.status == "cancelled"
    assert state.result is None
    assert RunMonitor(run.id, factory).beat() is False


def test_completed_result_survives_late_cancel(context):
    _, db, store, run, _ = context
    store.mark_completed(run.id, {"saved": True}, db=db)
    store.mark_cancelled(run.id, db=db)
    assert store.get(run.id, db=db).status == "completed"
    assert store.get(run.id, db=db).result == {"saved": True}


def make_client(db, store, user):
    app = FastAPI()
    app.include_router(runs.router)
    app.dependency_overrides[runs._get_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: user
    return TestClient(app)


def test_cancel_dispatches_termination_and_is_idempotent(context):
    _, db, store, run, user = context
    store.mark_running(run.id, db=db)
    with (
        make_client(db, store, user) as client,
        patch("api.routers.runs.request_cancel"),
        patch("api.worker.celery_app.control.revoke") as revoke,
    ):
        for _ in range(2):
            response = client.post(f"/runs/{run.id}/cancel")
            assert response.status_code == 200
            assert response.json()["status"] == "cancelled"
        revoke.assert_called_with("task-123", terminate=True, signal="SIGTERM")


def test_dispatch_failure_remains_retryable(context):
    _, db, store, run, user = context
    store.mark_running(run.id, db=db)
    with (
        make_client(db, store, user) as client,
        patch("api.routers.runs.request_cancel"),
        patch("api.worker.celery_app.control.revoke", side_effect=ConnectionError),
    ):
        assert client.post(f"/runs/{run.id}/cancel").status_code == 503
        assert store.get(run.id, db=db).status == "running"


def test_other_account_cannot_inspect_or_cancel(context):
    _, db, store, run, _ = context
    other = UserRow(id=987, email="other@example.com", password_hash="unused")
    with (
        make_client(db, store, other) as client,
        patch("api.routers.runs.request_cancel") as cancel,
    ):
        assert client.get(f"/runs/{run.id}").status_code == 404
        assert client.post(f"/runs/{run.id}/cancel").status_code == 404
        cancel.assert_not_called()


def test_worker_loss_records_failure_without_overwriting_cancel(context):
    factory, db, store, run, _ = context
    from api.worker import record_worker_failure

    with (
        patch("api.database.get_session_factory", return_value=factory),
        patch("api.progress.publish_sentinel"),
    ):
        record_worker_failure(
            task_id="task-123", exception=RuntimeError("WorkerLostError")
        )
        db.expire_all()
        assert store.get(run.id, db=db).status == "failed"


def test_liveness_migration_preserves_existing_runs():
    import importlib.util

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect, text

    spec = importlib.util.spec_from_file_location(
        "migration", "migrations/versions/0005_run_liveness.py"
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE runs (id TEXT PRIMARY KEY)"))
        connection.execute(text("INSERT INTO runs (id) VALUES ('existing')"))
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        assert connection.execute(
            text("SELECT progress_fraction, progress_message, heartbeat_at FROM runs")
        ).one() == (0, "", None)
        migration.downgrade()
        assert [
            column["name"] for column in inspect(connection).get_columns("runs")
        ] == ["id"]
        assert connection.execute(text("SELECT id FROM runs")).scalar() == "existing"
    engine.dispose()
