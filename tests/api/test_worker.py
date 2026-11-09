"""Tests for :mod:`api.worker` — Celery task with eager execution.

Uses ``task_always_eager = True`` so tasks run in-process.
Coordinator functions are mocked to avoid heavy computation.
Progress publisher and sentinel are mocked (no Redis in tests).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.db_store import DBRunStore
from api.models import Base, UserRow
from api.schemas import RunStatus, RunType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_cancellation_client():
    # The worker creates a separate cancellation client from its publisher.
    # Keep these coordinator tests offline even when checking cancellation flags.
    client = MagicMock()
    client.exists.return_value = 0
    with patch("redis.from_url", return_value=client):
        yield client


@pytest.fixture(autouse=True)
def mock_monitor_thread():
    with patch("api.run_monitor.RunMonitor.start"), patch(
        "api.run_monitor.RunMonitor.stop"
    ):
        yield


@pytest.fixture()
def db_session() -> Session:  # type: ignore[misc]
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
def user(db_session: Session) -> UserRow:
    u = UserRow(id=1, email="worker@test.com", password_hash="fakehash")
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


FAKE_RESULT: dict[str, Any] = {
    "portfolio_value": 9999.99,
    "weights": {"AAPL": 1.0},
}


def _mock_progress_publisher(
    run_id: str, redis_url: str | None = None
) -> tuple[MagicMock, Any]:
    """Return a (mock_redis_client, noop_callback) pair."""
    mock_client = MagicMock()

    def _noop(fraction: float, message: str) -> None:
        pass

    return mock_client, _noop


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunCoordinatorTask:
    """Test the Celery task with mocked coordinators and eager mode."""

    def test_execution_task_marks_completed(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """The task should call the coordinator and mark the run completed."""
        store = DBRunStore()
        run = store.create(
            RunType.EXECUTION,
            {"data_source": "alpaca"},
            db=db_session,
            user_id=user.id,
        )

        # We need to patch the session factory and the coordinator
        mock_factory = MagicMock(return_value=db_session)

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "investment_tool.app.execution_coordinator.execute",
                return_value=FAKE_RESULT,
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_mock_progress_publisher,
            ),
            patch("api.progress.publish_sentinel") as mock_sentinel,
        ):
            from api.worker import run_coordinator_task

            # Call the task function directly (not via .delay)
            result = run_coordinator_task(
                run.id, "execution", {"data_source": "alpaca"}
            )

        assert result["status"] == "completed"
        assert result["run_id"] == run.id

        updated = store.get(run.id, db=db_session)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETED
        assert updated.result is not None

        # Verify sentinel was published
        mock_sentinel.assert_called_once_with(run.id, status="completed")

    def test_execution_task_marks_failed_on_error(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """If the coordinator raises, the run should be marked failed."""
        store = DBRunStore()
        run = store.create(
            RunType.EXECUTION,
            {},
            db=db_session,
            user_id=user.id,
        )

        mock_factory = MagicMock(return_value=db_session)

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "investment_tool.app.execution_coordinator.execute",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_mock_progress_publisher,
            ),
            patch("api.progress.publish_sentinel") as mock_sentinel,
        ):
            from api.worker import run_coordinator_task

            result = run_coordinator_task(run.id, "execution", {})

        assert result["status"] == "failed"
        assert "boom" in result["error"]

        updated = store.get(run.id, db=db_session)
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert "boom" in (updated.error_message or "")

        # Verify failure sentinel was published
        mock_sentinel.assert_called_once_with(run.id, status="failed", error="boom")

    def test_strategy_comparison_task(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """Strategy comparison task should work end-to-end."""
        store = DBRunStore()
        run = store.create(
            RunType.STRATEGY_COMPARISON,
            {},
            db=db_session,
            user_id=user.id,
        )

        mock_factory = MagicMock(return_value=db_session)

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "investment_tool.app.strategy_comparison_coordinator.compare_strategies",
                return_value=FAKE_RESULT,
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_mock_progress_publisher,
            ),
            patch("api.progress.publish_sentinel"),
        ):
            from api.worker import run_coordinator_task

            result = run_coordinator_task(run.id, "strategy_comparison", {})

        assert result["status"] == "completed"

    def test_failed_comparison_exposes_completed_periods(
        self, db_session: Session, user: UserRow
    ) -> None:
        store = DBRunStore()
        run = store.create(
            RunType.STRATEGY_COMPARISON, {}, db=db_session, user_id=user.id
        )

        def fail_after_checkpoint(*args: Any, **kwargs: Any) -> None:
            partial = kwargs["partial_sink"]
            partial["individual_results"] = {
                "optimization": {"results": [{"total_value": 1001.0}]}
            }
            kwargs["partial_callback"](partial)
            raise RuntimeError("later strategy failed")

        with (
            patch(
                "api.database.get_session_factory",
                return_value=MagicMock(return_value=db_session),
            ),
            patch(
                "investment_tool.app.strategy_comparison_coordinator.compare_strategies",
                side_effect=fail_after_checkpoint,
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_mock_progress_publisher,
            ),
            patch("api.progress.publish_sentinel"),
        ):
            from api.worker import run_coordinator_task

            result = run_coordinator_task(run.id, "strategy_comparison", {})

        assert result["status"] == "failed"
        saved = store.get(run.id, db=db_session)
        assert saved is not None and saved.result is not None
        assert saved.result["individual_results"]["optimization"]["results"] == [
            {"total_value": 1001.0}
        ]

    def test_unknown_run_type_marks_failed(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """An unknown run_type should mark the run as failed."""
        store = DBRunStore()
        run = store.create(
            RunType.EXECUTION,  # type doesn't matter for the store
            {},
            db=db_session,
            user_id=user.id,
        )

        mock_factory = MagicMock(return_value=db_session)

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_mock_progress_publisher,
            ),
            patch("api.progress.publish_sentinel"),
        ):
            from api.worker import run_coordinator_task

            result = run_coordinator_task(run.id, "nonexistent_type", {})

        assert result["status"] == "failed"
        assert "Unknown run_type" in result["error"]

    def test_progress_callback_passed_to_coordinator(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """The progress callback should be passed to the coordinator."""
        store = DBRunStore()
        run = store.create(
            RunType.EXECUTION,
            {"data_source": "alpaca"},
            db=db_session,
            user_id=user.id,
        )

        mock_factory = MagicMock(return_value=db_session)
        mock_coordinator = MagicMock(return_value=FAKE_RESULT)

        captured_callback = None

        def _capture_publisher(
            run_id: str, redis_url: str | None = None
        ) -> tuple[MagicMock, Any]:
            nonlocal captured_callback
            mock_client = MagicMock()

            def _cb(fraction: float, message: str) -> None:
                pass

            captured_callback = _cb
            return mock_client, _cb

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "investment_tool.app.execution_coordinator.execute",
                mock_coordinator,
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=_capture_publisher,
            ),
            patch("api.progress.publish_sentinel"),
        ):
            from api.worker import run_coordinator_task

            run_coordinator_task(run.id, "execution", {"data_source": "alpaca"})

        # The coordinator should have been called with (params, callback)
        assert mock_coordinator.call_count == 1
        args = mock_coordinator.call_args
        assert len(args[0]) == 2  # positional: (params, progress_callback)
        # The worker wraps the raw publisher callback in a cancellable wrapper;
        # it should be a callable (not necessarily the same object).
        assert callable(args[0][1])

    def test_progress_publisher_failure_does_not_block(
        self,
        db_session: Session,
        user: UserRow,
    ) -> None:
        """If make_progress_publisher fails, the task should still run."""
        store = DBRunStore()
        run = store.create(
            RunType.EXECUTION,
            {"data_source": "alpaca"},
            db=db_session,
            user_id=user.id,
        )

        mock_factory = MagicMock(return_value=db_session)

        with (
            patch("api.database.get_session_factory", return_value=mock_factory),
            patch(
                "investment_tool.app.execution_coordinator.execute",
                return_value=FAKE_RESULT,
            ),
            patch(
                "api.progress.make_progress_publisher",
                side_effect=ConnectionError("Redis down"),
            ),
            patch("api.progress.publish_sentinel"),
        ):
            from api.worker import run_coordinator_task

            result = run_coordinator_task(
                run.id, "execution", {"data_source": "alpaca"}
            )

        # Task should still complete — progress is best-effort
        assert result["status"] == "completed"


def test_readiness_failure_keeps_report_when_no_period_completed(db_session, user):
    from investment_tool.exceptions import BacktestReadinessError
    from api.worker import run_coordinator_task

    report = {
        "issues": [{"symbol": "RETA", "action": "block_backtest"}],
        "readiness": {"ready": False, "affected_symbols": ["RETA"]},
    }
    store = DBRunStore()
    run = store.create(RunType.EXECUTION, {}, db=db_session, user_id=user.id)
    with (
        patch(
            "api.database.get_session_factory",
            return_value=MagicMock(return_value=db_session),
        ),
        patch(
            "investment_tool.app.execution_coordinator.execute",
            side_effect=BacktestReadinessError(report),
        ),
        patch(
            "api.progress.make_progress_publisher", side_effect=_mock_progress_publisher
        ),
        patch("api.progress.publish_sentinel"),
    ):
        result = run_coordinator_task(run.id, "execution", {})
    assert result["status"] == "failed"
    updated = store.get(run.id, db=db_session)
    assert updated.result["validation_report"] == report
    assert "before optimization" in updated.error_message
