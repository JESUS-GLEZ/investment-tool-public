"""Campaign metadata is private, optional and independent of execution."""

import importlib.util
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.auth import get_current_user
from api.database import get_db
from api.db_store import DBRunStore
from api.models import Base, UserRow
from api.routers import campaigns, comparisons, runs
from api.schemas import RunType


@pytest.fixture()
def campaign_client(monkeypatch):
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        users = [
            UserRow(email=f"campaign{i}@example.com", password_hash="unused")
            for i in range(2)
        ]
        db.add_all(users)
        db.commit()
        current = [users[0]]
        store = DBRunStore()
        runs.set_store(store)
        comparisons.set_store(store)
        app = FastAPI()
        for router in (campaigns.router, runs.router, comparisons.router):
            app.include_router(router)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: current[0]
        task = MagicMock()
        task.id = "test-task"
        delay = MagicMock(return_value=task)
        monkeypatch.setattr("api.worker.run_coordinator_task.delay", delay)
        for name in ("strategy", "solver", "prediction"):
            monkeypatch.setattr(
                f"api.adapters.build_{name}_comparison_params", lambda *_: object()
            )
        with TestClient(app) as client:
            yield client, db, current, users, store, delay
    engine.dispose()


def test_campaign_create_edit_counts_and_account_isolation(campaign_client):
    client, db, current, users, store, delay = campaign_client
    response = client.post(
        "/campaigns",
        json={
            "name": "  Costs  ",
            "hypothesis": "Fewer trades help",
            "notes": "Compare years",
        },
    )
    assert response.status_code == 201
    saved = response.json()
    assert saved["name"] == "Costs" and saved["run_count"] == 0
    run = store.create(
        RunType.EXECUTION, {"initial_cash": 1000}, db=db, user_id=users[0].id
    )
    store.mark_completed(run.id, {"value": 1200}, db=db)
    before = client.get(f"/runs/{run.id}").json()
    url = f"/runs/{run.id}/campaign"
    assert (
        client.put(url, json={"campaign_id": saved["id"]}).json()["campaign_id"]
        == saved["id"]
    )
    assert client.get("/campaigns").json()[0]["run_count"] == 1
    after = client.get(f"/runs/{run.id}").json()
    assert {k: v for k, v in before.items() if k != "campaign_id"} == {
        k: v for k, v in after.items() if k != "campaign_id"
    }
    assert not delay.called
    current[0] = users[1]
    assert client.get("/campaigns").json() == []
    assert (
        client.put(f"/campaigns/{saved['id']}", json={"name": "Stolen"}).status_code
        == 404
    )
    assert client.put(url, json={"campaign_id": None}).status_code == 404
    own_run = store.create(RunType.EXECUTION, {}, db=db, user_id=users[1].id)
    assert (
        client.put(
            f"/runs/{own_run.id}/campaign", json={"campaign_id": saved["id"]}
        ).status_code
        == 404
    )
    assert client.post("/runs", json={"campaign_id": saved["id"]}).status_code == 404
    current[0] = users[0]
    updated = client.put(
        f"/campaigns/{saved['id']}",
        json={"name": "Renamed", "hypothesis": "Revised", "notes": "Finding"},
    ).json()
    assert updated["name"] == "Renamed" and updated["run_count"] == 1
    assert client.get("/runs").json()[0]["campaign_id"] == saved["id"]
    assert client.put(url, json={"campaign_id": None}).json()["campaign_id"] is None
    assert client.get("/campaigns").json()[0]["run_count"] == 0


@pytest.mark.parametrize(
    "endpoint",
    [
        "/runs",
        "/comparisons/strategy",
        "/comparisons/solver",
        "/comparisons/prediction",
    ],
)
def test_all_run_types_can_be_grouped_without_passing_campaign_to_engine(
    campaign_client, endpoint
):
    client, _, _, _, _, delay = campaign_client
    cid = client.post("/campaigns", json={"name": "Research"}).json()["id"]
    response = client.post(endpoint, json={"campaign_id": cid})
    assert response.status_code == 202, response.text
    assert response.json()["campaign_id"] == cid
    run = client.get(f"/runs/{response.json()['id']}").json()
    assert run["campaign_id"] == cid
    assert "campaign_id" not in run["request_payload"]
    assert "campaign_id" not in delay.call_args.args[2]


@pytest.mark.parametrize(
    "body",
    [
        {"name": " "},
        {"name": "x" * 121},
        {"name": "x", "notes": "x" * 10001},
        {"name": "x", "user_id": 2},
    ],
)
def test_invalid_campaign_metadata_is_rejected(campaign_client, body):
    assert campaign_client[0].post("/campaigns", json=body).status_code == 422


def test_campaign_migration_preserves_existing_runs():
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    spec = importlib.util.spec_from_file_location(
        "campaign_migration", "migrations/versions/0007_campaigns.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE users (id INTEGER PRIMARY KEY)"))
        conn.execute(
            text(
                "CREATE TABLE runs (id VARCHAR(12) PRIMARY KEY, user_id INTEGER, request_payload TEXT)"
            )
        )
        conn.execute(text("INSERT INTO runs VALUES ('existing', 1, 'preserved')"))
        module.op = Operations(MigrationContext.configure(conn))
        module.upgrade()
        assert conn.execute(
            text("SELECT id, request_payload, campaign_id FROM runs")
        ).one() == ("existing", "preserved", None)
        assert "campaigns" in inspect(conn).get_table_names()
        module.downgrade()
        assert conn.execute(text("SELECT id, request_payload FROM runs")).one() == (
            "existing",
            "preserved",
        )
        assert "campaigns" not in inspect(conn).get_table_names()
    engine.dispose()


def test_assignment_locks_only_runs_with_joined_results(campaign_client):
    """PostgreSQL forbids an unscoped lock on a nullable outer-joined result."""
    from sqlalchemy import event
    from sqlalchemy.dialects import postgresql

    client, db, _, users, store, _ = campaign_client
    run = store.create(RunType.EXECUTION, {}, db=db, user_id=users[0].id)
    statements = []

    def capture(execution):
        statement = execution.statement
        if getattr(statement, "_for_update_arg", None) is not None:
            statements.append(str(statement.compile(dialect=postgresql.dialect())))

    event.listen(db, "do_orm_execute", capture)
    try:
        assert (
            client.put(
                f"/runs/{run.id}/campaign", json={"campaign_id": None}
            ).status_code
            == 200
        )
    finally:
        event.remove(db, "do_orm_execute", capture)
    assert len(statements) == 1
    assert statements[0].endswith("FOR UPDATE OF runs")
