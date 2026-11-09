"""Tests for :mod:`api.store` — in-memory RunStore."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from api.schemas import RunStatus, RunType
from api.store import RunStore


class TestRunStoreCreate:
    def test_create_returns_queued_run(self) -> None:
        store = RunStore()
        run = store.create(RunType.EXECUTION, {"key": "val"})
        assert run.status == RunStatus.QUEUED
        assert run.run_type == RunType.EXECUTION
        assert run.request_payload == {"key": "val"}
        assert run.result is None
        assert run.error_message is None
        assert run.completed_at is None

    def test_create_generates_unique_ids(self) -> None:
        store = RunStore()
        ids = {store.create(RunType.EXECUTION, {}).id for _ in range(50)}
        assert len(ids) == 50


class TestRunStoreLifecycle:
    def test_mark_running(self) -> None:
        store = RunStore()
        run = store.create(RunType.EXECUTION, {})
        store.mark_running(run.id)
        updated = store.get(run.id)
        assert updated is not None
        assert updated.status == RunStatus.RUNNING

    def test_mark_completed(self) -> None:
        store = RunStore()
        run = store.create(RunType.EXECUTION, {})
        store.mark_running(run.id)
        store.mark_completed(run.id, {"sharpe": 1.5})
        updated = store.get(run.id)
        assert updated is not None
        assert updated.status == RunStatus.COMPLETED
        assert updated.result == {"sharpe": 1.5}
        assert updated.completed_at is not None

    def test_mark_failed(self) -> None:
        store = RunStore()
        run = store.create(RunType.EXECUTION, {})
        store.mark_running(run.id)
        store.mark_failed(run.id, "boom")
        updated = store.get(run.id)
        assert updated is not None
        assert updated.status == RunStatus.FAILED
        assert updated.error_message == "boom"
        assert updated.completed_at is not None


class TestRunStoreRead:
    def test_get_returns_none_for_missing(self) -> None:
        store = RunStore()
        assert store.get("nonexistent") is None

    def test_list_all_sorted_newest_first(self) -> None:
        store = RunStore()
        r1 = store.create(RunType.EXECUTION, {})
        r2 = store.create(RunType.STRATEGY_COMPARISON, {})
        r3 = store.create(RunType.SOLVER_COMPARISON, {})
        runs = store.list_all()
        ids = [r.id for r in runs]
        # Newest first
        assert ids == [r3.id, r2.id, r1.id]

    def test_clear(self) -> None:
        store = RunStore()
        store.create(RunType.EXECUTION, {})
        store.create(RunType.EXECUTION, {})
        assert len(store.list_all()) == 2
        store.clear()
        assert len(store.list_all()) == 0


class TestRunStoreDeletion:
    def test_active_run_and_cleanup_failure_preserve_record(self):
        store = RunStore()
        run = store.create(RunType.EXECUTION, {})
        cleanup = Mock()
        with pytest.raises(ValueError):
            store.delete(run.id, cleanup=cleanup)
        cleanup.assert_not_called()
        store.mark_failed(run.id, "failed", {"partial": True})
        cleanup.side_effect = OSError("disk error")
        with pytest.raises(OSError):
            store.delete(run.id, cleanup=cleanup)
        assert store.get(run.id).result == {"partial": True}
        cleanup.side_effect = None
        store.delete(run.id, cleanup=cleanup)
        assert store.get(run.id) is None
