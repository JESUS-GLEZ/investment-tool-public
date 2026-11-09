"""Tests for :mod:`api.db_store` — database-backed RunStore.

Uses an in-memory SQLite database for isolation (no Postgres needed).
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.db_store import DBRunStore
from api.models import Base, UserRow
from api.schemas import RunStatus, RunType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db() -> Session:  # type: ignore[misc]
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    session = factory()
    try:
        yield session  # type: ignore[misc]
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def user(db: Session) -> UserRow:
    """Create and return a test user."""
    u = UserRow(id=1, email="test@test.com", password_hash="fakehash")
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@pytest.fixture()
def store() -> DBRunStore:
    return DBRunStore()


# ---------------------------------------------------------------------------
# Tests — Create
# ---------------------------------------------------------------------------


class TestDBRunStoreCreate:
    def test_create_returns_queued_run(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        run = store.create(
            RunType.EXECUTION,
            {"key": "val"},
            db=db,
            user_id=user.id,
        )
        assert run.status == RunStatus.QUEUED
        assert run.run_type == RunType.EXECUTION
        assert run.request_payload == {"key": "val"}
        assert run.result is None
        assert run.error_message is None
        assert run.completed_at is None

    def test_create_generates_unique_ids(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        ids = {
            store.create(RunType.EXECUTION, {}, db=db, user_id=user.id).id
            for _ in range(50)
        }
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# Tests — Lifecycle
# ---------------------------------------------------------------------------


class TestDBRunStoreLifecycle:
    def test_mark_running(self, store: DBRunStore, db: Session, user: UserRow) -> None:
        run = store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.mark_running(run.id, db=db)
        updated = store.get(run.id, db=db)
        assert updated is not None
        assert updated.status == RunStatus.RUNNING

    def test_mark_completed(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        run = store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.mark_running(run.id, db=db)
        store.mark_completed(run.id, {"sharpe": 1.5}, db=db)
        updated = store.get(run.id, db=db)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETED
        assert updated.result == {"sharpe": 1.5}
        assert updated.completed_at is not None

    def test_mark_failed(self, store: DBRunStore, db: Session, user: UserRow) -> None:
        run = store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.mark_running(run.id, db=db)
        store.mark_failed(run.id, "boom", db=db)
        updated = store.get(run.id, db=db)
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert updated.error_message == "boom"
        assert updated.completed_at is not None

    def test_failed_run_retains_last_checkpoint(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        run = store.create(RunType.STRATEGY_COMPARISON, {}, db=db, user_id=user.id)
        store.mark_running(run.id, db=db)
        store.save_partial(run.id, {"periods": [1]}, db=db)
        store.save_partial(run.id, {"periods": [1, 2]}, db=db)
        store.mark_failed(run.id, "solver failed", db=db)
        updated = store.get(run.id, db=db)
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert updated.result == {"periods": [1, 2]}

    def test_set_celery_task_id(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        run = store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.set_celery_task_id(run.id, "celery-123", db=db)
        from api.models import RunRow

        row = db.query(RunRow).filter(RunRow.id == run.id).first()
        assert row is not None
        assert row.celery_task_id == "celery-123"


# ---------------------------------------------------------------------------
# Tests — Read
# ---------------------------------------------------------------------------


class TestDBRunStoreRead:
    def test_get_returns_none_for_missing(self, store: DBRunStore, db: Session) -> None:
        assert store.get("nonexistent", db=db) is None

    def test_list_all_sorted_newest_first(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        r1 = store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        r2 = store.create(RunType.STRATEGY_COMPARISON, {}, db=db, user_id=user.id)
        r3 = store.create(RunType.SOLVER_COMPARISON, {}, db=db, user_id=user.id)
        runs = store.list_all(db=db, user_id=user.id)
        ids = [r.id for r in runs]
        # Newest first
        assert ids == [r3.id, r2.id, r1.id]

    def test_list_all_scoped_to_user(
        self, store: DBRunStore, db: Session, user: UserRow
    ) -> None:
        # Create a second user
        u2 = UserRow(id=2, email="other@test.com", password_hash="fakehash2")
        db.add(u2)
        db.commit()
        db.refresh(u2)

        store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.create(RunType.EXECUTION, {}, db=db, user_id=u2.id)

        runs_u1 = store.list_all(db=db, user_id=user.id)
        runs_u2 = store.list_all(db=db, user_id=u2.id)
        assert len(runs_u1) == 1
        assert len(runs_u2) == 1

    def test_clear(self, store: DBRunStore, db: Session, user: UserRow) -> None:
        store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        store.create(RunType.EXECUTION, {}, db=db, user_id=user.id)
        assert len(store.list_all(db=db)) == 2
        store.clear(db=db)
        assert len(store.list_all(db=db)) == 0


def test_list_has_saved_settings_without_loading_results(
    store: DBRunStore, db: Session, user: UserRow
) -> None:
    from sqlalchemy import event

    payload = {"strategy_type": "optimization", "population_size": 200}
    run = store.create(RunType.EXECUTION, payload, db=db, user_id=user.id)
    store.mark_completed(run.id, {"optimal_weights": {"AAPL": 1}}, db=db)
    db.expunge_all()
    statements: list[str] = []

    def capture(_conn, _cursor, statement, _parameters, _context, _executemany):
        statements.append(statement)

    event.listen(db.get_bind(), "before_cursor_execute", capture)
    try:
        summary = store.list_all(db=db, user_id=user.id)[0]
        assert summary.request_payload == payload
        assert "result" not in summary.model_dump()
        assert len(statements) == 1
        assert "run_results" not in statements[0].lower()
        detail = store.get(run.id, db=db)
        assert detail is not None
        assert detail.result == {"optimal_weights": {"AAPL": 1}}
    finally:
        event.remove(db.get_bind(), "before_cursor_execute", capture)
