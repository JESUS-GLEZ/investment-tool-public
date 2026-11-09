"""Inverted asset bounds are rejected before run or preset side effects."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from api.auth import get_current_user
from api.database import get_db
from api.main import create_app
from api.models import Base, PresetRow, UserRow
from api.store import RunStore

CASES = [
    ("/runs", "execution", "max_assets"),
    ("/comparisons/solver", "solver_comparison", "num_assets"),
    # Preserve the already-enforced strategy-comparison rule as a control.
    ("/comparisons/strategy", "strategy_comparison", "max_assets"),
]


@pytest.fixture()
def bounds_client():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        user = UserRow(email="synthetic@example.test", password_hash="unused")
        db.add(user)
        db.commit()
        store = RunStore()
        store.create = Mock(side_effect=AssertionError("invalid request created a run"))
        app = create_app(store=store)
        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_db] = lambda: db
        with TestClient(app) as client:
            yield client, store, db
    engine.dispose()


def invalid_bounds(case, upper):
    return [
        {"min_assets": 6, upper: 5},
        {"min_assets": 6},
        {upper: 1},
    ][case]


@pytest.mark.parametrize(("endpoint", "run_type", "upper"), CASES)
@pytest.mark.parametrize("case", range(3))
def test_inverted_bounds_do_not_create_run(
    bounds_client, endpoint, run_type, upper, case
):
    client, store, _ = bounds_client
    response = client.post(endpoint, json=invalid_bounds(case, upper))
    assert response.status_code == 422
    assert "Minimum assets cannot exceed" in response.json()["detail"][0]["msg"]
    store.create.assert_not_called()


@pytest.mark.parametrize(("endpoint", "run_type", "upper"), CASES)
@pytest.mark.parametrize("case", range(3))
def test_invalid_preset_create_and_update_preserve_valid_equal_bounds(
    bounds_client, endpoint, run_type, upper, case
):
    client, store, db = bounds_client
    body = {
        "name": "owned",
        "run_type": run_type,
        "parameters": {"min_assets": 3, upper: 3},
    }
    created = client.post("/presets", json=body)
    assert created.status_code == 201
    saved = created.json()
    assert saved["parameters"]["min_assets"] == saved["parameters"][upper] == 3
    invalid = {**body, "name": "invalid", "parameters": invalid_bounds(case, upper)}
    for method, url in [
        (client.post, "/presets"),
        (client.put, f"/presets/{saved['id']}"),
    ]:
        response = method(url, json=invalid)
        assert response.status_code == 422
        assert "Minimum assets cannot exceed" in response.json()["detail"][0]["msg"]
    assert db.query(PresetRow).count() == 1
    assert client.get("/presets", params={"run_type": run_type}).json() == [saved]
    store.create.assert_not_called()


@pytest.mark.parametrize(("endpoint", "run_type", "upper"), CASES)
def test_default_presets_keep_existing_bounds(bounds_client, endpoint, run_type, upper):
    client, _, _ = bounds_client
    response = client.post(
        "/presets", json={"name": "defaults", "run_type": run_type, "parameters": {}}
    )
    assert response.status_code == 201
    assert response.json()["parameters"]["min_assets"] == 2
    assert response.json()["parameters"][upper] == 5


@pytest.mark.parametrize(("endpoint", "run_type", "upper"), CASES[:2])
@pytest.mark.parametrize("recovery", ["correct", "delete"])
def test_legacy_inverted_presets_remain_listable_and_recoverable(
    bounds_client, endpoint, run_type, upper, recovery
):
    client, store, db = bounds_client
    body = {
        "name": "valid",
        "run_type": run_type,
        "parameters": {"min_assets": 3, upper: 3},
    }
    valid = client.post("/presets", json=body).json()
    # Seed a row in the exact form accepted by the previous request schema,
    # bypassing today's write validation as a pre-upgrade database would.
    parameters = {**valid["parameters"], "min_assets": 6, upper: 5}
    legacy = PresetRow(
        user_id=db.query(UserRow).one().id,
        name="legacy",
        run_type=run_type,
        parameters=parameters,
    )
    db.add(legacy)
    db.commit()
    url = f"/presets/{legacy.id}"
    response = client.get("/presets", params={"run_type": run_type})
    assert response.status_code == 200
    rows = {row["id"]: row for row in response.json()}
    assert rows[valid["id"]] == valid
    assert rows[legacy.id]["parameters"] == parameters
    invalid = {"name": "legacy", "run_type": run_type, "parameters": parameters}
    assert client.put(url, json=invalid).status_code == 422
    assert client.post(endpoint, json=parameters).status_code == 422
    db.refresh(legacy)
    assert legacy.parameters == parameters
    if recovery == "correct":
        corrected = {**parameters, "min_assets": 3, upper: 3}
        response = client.put(url, json={**invalid, "parameters": corrected})
        assert response.status_code == 200
        assert response.json()["parameters"] == corrected
    else:
        assert client.delete(url).status_code == 204
        assert client.get("/presets", params={"run_type": run_type}).json() == [valid]
    store.create.assert_not_called()
