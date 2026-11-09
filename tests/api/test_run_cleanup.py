from unittest.mock import Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import get_current_user
from api.database import get_db
from api.models import RunResultRow, UserRow
from api.routers import runs
from api.run_cleanup import delete_run_outputs, ensure_cancelled_task_stopped
from tests.api import test_run_lifecycle

context = test_run_lifecycle.context


def make_client(context, monkeypatch, tmp_path, other_user=False):
    _, db, store, _, user = context
    app = FastAPI()
    app.include_router(runs.router)
    runs.set_store(store)
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_current_user] = lambda: (
        UserRow(id=user.id + 1) if other_user else user
    )
    monkeypatch.setattr(runs, "_OUTPUT_BASE", tmp_path / "outputs")
    return TestClient(app)


def test_deletion_removes_owned_result_and_outputs_but_keeps_shared_data(
    context, monkeypatch, tmp_path
):
    _, db, store, run, _ = context
    store.mark_completed(run.id, {"results": [1]}, db=db)
    folder = tmp_path / "outputs" / run.id
    folder.mkdir(parents=True)
    (folder / "checkpoint.json").write_text("large results")
    shared = tmp_path / "input.pkl"
    shared.write_text("shared inputs")
    sibling = tmp_path / "outputs" / "another-run"
    sibling.mkdir()
    client = make_client(context, monkeypatch, tmp_path)
    assert client.delete(f"/runs/{run.id}").status_code == 204
    assert store.get(run.id, db=db) is None
    assert db.query(RunResultRow).count() == 0
    assert not folder.exists()
    assert shared.exists() and sibling.exists()
    assert client.delete(f"/runs/{run.id}").status_code == 404


@pytest.mark.parametrize("running", [False, True])
def test_active_run_cannot_be_deleted(context, monkeypatch, tmp_path, running):
    _, db, store, run, _ = context
    if running:
        store.mark_running(run.id, db=db)
    client = make_client(context, monkeypatch, tmp_path)
    with patch("api.routers.runs.delete_run_outputs") as cleanup:
        assert client.delete(f"/runs/{run.id}").status_code == 409
        cleanup.assert_not_called()
    assert store.get(run.id, db=db) is not None


def test_other_account_cannot_delete_or_probe_run(context, monkeypatch, tmp_path):
    _, db, store, run, _ = context
    store.mark_completed(run.id, {}, db=db)
    client = make_client(context, monkeypatch, tmp_path, other_user=True)
    with patch("api.routers.runs.delete_run_outputs") as cleanup:
        assert client.delete(f"/runs/{run.id}").status_code == 404
        cleanup.assert_not_called()
    assert store.get(run.id, db=db) is not None


def test_filesystem_failure_keeps_record_and_result_for_retry(
    context, monkeypatch, tmp_path
):
    _, db, store, run, _ = context
    store.mark_completed(run.id, {"valuable": True}, db=db)
    client = make_client(context, monkeypatch, tmp_path)
    with patch(
        "api.routers.runs.delete_run_outputs", side_effect=OSError("disk error")
    ):
        assert client.delete(f"/runs/{run.id}").status_code == 503
    assert store.get(run.id, db=db).result == {"valuable": True}


def test_cancelled_run_requires_confirmed_worker_stop(context, monkeypatch, tmp_path):
    _, db, store, run, _ = context
    store.mark_cancelled(run.id, db=db)
    client = make_client(context, monkeypatch, tmp_path)
    with patch(
        "api.routers.runs.ensure_cancelled_task_stopped",
        side_effect=ValueError("still active"),
    ), patch("api.routers.runs.delete_run_outputs") as cleanup:
        assert client.delete(f"/runs/{run.id}").status_code == 409
        cleanup.assert_not_called()
    with patch(
        "api.routers.runs.ensure_cancelled_task_stopped",
        side_effect=RuntimeError("offline"),
    ):
        assert client.delete(f"/runs/{run.id}").status_code == 503
    with patch("api.routers.runs.ensure_cancelled_task_stopped") as stopped:
        assert client.delete(f"/runs/{run.id}").status_code == 204
        stopped.assert_called_once_with("task-123")


@pytest.mark.parametrize(
    "mode",
    [
        "active",
        "reserved",
        "scheduled",
        "offline",
        "changed_nodes",
        "broker_error",
        "stopped",
    ],
)
def test_cancelled_worker_check(mode):
    inspector = Mock()
    for method in ("active", "reserved", "scheduled"):
        getattr(inspector, method).return_value = {"node": []}
    if mode == "offline":
        inspector.active.return_value = None
    elif mode == "changed_nodes":
        inspector.reserved.return_value = {"other-node": []}
    elif mode == "broker_error":
        inspector.active.side_effect = ConnectionError("broker offline")
    elif mode != "stopped":
        task = {"id": "task"} if mode != "scheduled" else {"request": {"id": "task"}}
        getattr(inspector, mode).return_value = {"node": [task]}
    with patch("api.worker.celery_app.control.inspect", return_value=inspector):
        if mode in {"offline", "changed_nodes", "broker_error"}:
            with pytest.raises(RuntimeError):
                ensure_cancelled_task_stopped("task")
        elif mode == "stopped":
            ensure_cancelled_task_stopped("task")
        else:
            with pytest.raises(ValueError):
                ensure_cancelled_task_stopped("task")


def test_output_cleanup_refuses_traversal_and_symlink(tmp_path):
    with pytest.raises(ValueError):
        delete_run_outputs(tmp_path, "../outside")
    target = tmp_path / "external"
    target.mkdir()
    (target / "keep").write_text("keep")
    run_path = tmp_path / "aaaaaaaaaaaa"
    run_path.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        delete_run_outputs(tmp_path, run_path.name)
    assert (target / "keep").exists()


def test_output_cleanup_is_retryable_when_directory_is_already_missing(tmp_path):
    delete_run_outputs(tmp_path, "aaaaaaaaaaaa")
    assert list(tmp_path.iterdir()) == []
