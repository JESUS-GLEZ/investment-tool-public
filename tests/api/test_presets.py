"""Preset persistence, request validation and account isolation."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.auth import get_current_user
from api.database import get_db
from api.models import Base, UserRow
from api.routers.presets import router
from api.schemas import (
    ExecutionRequest,
    PredictionComparisonRequest,
    SolverComparisonRequest,
    StrategyComparisonRequest,
)


@pytest.fixture()
def preset_client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        users = [
            UserRow(email=f"user{i}@example.com", password_hash="unused")
            for i in range(2)
        ]
        db.add_all(users)
        db.commit()
        current = [users[0]]
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_db] = lambda: db
        app.dependency_overrides[get_current_user] = lambda: current[0]
        with TestClient(app) as client:
            yield client, current, users
    engine.dispose()


def test_crud_and_isolation(preset_client):
    client, current, users = preset_client
    body = {
        "name": " Local comparison ",
        "run_type": "solver_comparison",
        "parameters": {
            "data_source": "local",
            "input_file": "prices.pkl",
            "solvers": ["GWASFGA"],
            "historical_start_date": "2021-01-01",
            "historical_end_date": "2024-12-31",
        },
    }
    response = client.post("/presets", json=body)
    assert response.status_code == 201
    saved = response.json()
    assert saved["name"] == "Local comparison"
    assert saved["parameters"]["historical_start_date"] == "2021-01-01"
    url = f"/presets/{saved['id']}"
    assert len(client.get("/presets?run_type=solver_comparison").json()) == 1
    assert client.get("/presets?run_type=execution").json() == []
    assert client.post("/presets", json=body).status_code == 409
    current[0] = users[1]
    assert client.get("/presets?run_type=solver_comparison").json() == []
    assert client.put(url, json=body).status_code == 404
    assert client.delete(url).status_code == 404
    current[0] = users[0]
    body["parameters"]["n_generations"] = 1234
    assert client.put(url, json=body).json()["parameters"]["n_generations"] == 1234
    assert client.delete(url).status_code == 204
    assert client.get("/presets?run_type=solver_comparison").json() == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"data_source": "local"},
        {"n_generations": 0},
        {"unknown_option": 1},
        {"historical_start_date": "2027-01-01", "historical_end_date": "2026-12-31"},
    ],
)
def test_rejects_invalid_presets(preset_client, parameters):
    client, _, _ = preset_client
    assert (
        client.post(
            "/presets",
            json={
                "name": "Invalid",
                "run_type": "solver_comparison",
                "parameters": parameters,
            },
        ).status_code
        == 422
    )


@pytest.mark.parametrize(
    "schema",
    [
        ExecutionRequest,
        StrategyComparisonRequest,
        SolverComparisonRequest,
        PredictionComparisonRequest,
    ],
)
@pytest.mark.parametrize(
    "start,end",
    [("2024-02-30", "2024-03-01"), ("2025-01-01", "2024-01-01"), ("", "2024-01-01")],
)
def test_invalid_historical_window(schema, start, end):
    with pytest.raises(ValueError):
        schema(historical_start_date=start, historical_end_date=end)


def test_migration_creates_and_removes_preset_table():
    import importlib.util

    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from sqlalchemy import inspect

    spec = importlib.util.spec_from_file_location(
        "preset_migration", "migrations/versions/0004_presets.py"
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        migration.op = Operations(MigrationContext.configure(connection))
        migration.upgrade()
        assert "presets" in inspect(connection).get_table_names()
        columns = {
            column["name"] for column in inspect(connection).get_columns("presets")
        }
        assert columns == {"id", "user_id", "run_type", "name", "parameters"}
        migration.downgrade()
        assert "presets" not in inspect(connection).get_table_names()
    engine.dispose()


def test_solver_seed_settings_survive_preset_roundtrip(preset_client):
    client, _, _ = preset_client
    settings = {
        "use_multi_greedy_seeds": False,
        "seed_max_subsets": 123,
        "seed_beam_width": 3,
        "include_markowitz_reference": True,
    }
    response = client.post(
        "/presets",
        json={
            "name": "Seed experiment",
            "run_type": "solver_comparison",
            "parameters": settings,
        },
    )
    assert response.status_code == 201
    saved = client.get("/presets?run_type=solver_comparison").json()[0]["parameters"]
    for key, value in settings.items():
        assert saved[key] == value


@pytest.mark.parametrize(
    "schema",
    [
        ExecutionRequest,
        StrategyComparisonRequest,
        SolverComparisonRequest,
        PredictionComparisonRequest,
    ],
)
def test_omitted_cutoff_uses_request_time_but_explicit_cutoffs_are_preserved(
    schema, monkeypatch
):
    from datetime import datetime, timezone
    import api.schemas as schemas

    class Clock(datetime):
        current = datetime(2026, 10, 27, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            return cls.current

    monkeypatch.setattr(schemas, "datetime", Clock)
    assert schema().historical_end_date == "2026-10-27"
    Clock.current = datetime(2027, 1, 1, tzinfo=timezone.utc)
    assert schema().historical_end_date == "2027-01-01"
    assert schema(historical_end_date="2024-06-30").historical_end_date == "2024-06-30"
