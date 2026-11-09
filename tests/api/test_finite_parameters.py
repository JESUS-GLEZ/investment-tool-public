"""Invalid numeric run inputs fail before execution or preset persistence."""

import json
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

RUNS = [
    ("/runs", "execution"),
    ("/comparisons/strategy", "strategy_comparison"),
    ("/comparisons/solver", "solver_comparison"),
    ("/comparisons/prediction", "prediction_comparison"),
]
INVALID_NUMBERS = [
    "Infinity",
    "-Infinity",
    "NaN",
    '"Infinity"',
    '"-Infinity"',
    '"NaN"',
    "1e999",
]
FIELDS = ["initial_cash", "transaction_cost_rate", "risk_free_rate"]


@pytest.fixture()
def numeric_client():
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


@pytest.mark.parametrize(("endpoint", "run_type"), RUNS)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("number", INVALID_NUMBERS)
def test_nonfinite_run_inputs_return_field_error_without_creating_run(
    numeric_client, endpoint, run_type, field, number
):
    client, store, _ = numeric_client
    response = client.post(
        endpoint,
        content=f'{{"{field}":{number}}}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors[0]["loc"] == ["body", field]
    assert errors[0]["type"] == "finite_number"
    # The response must contain no nonstandard JSON constants, including input.
    json.loads(response.text, parse_constant=lambda value: pytest.fail(value))
    store.create.assert_not_called()


@pytest.mark.parametrize(("endpoint", "run_type"), RUNS)
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("number", INVALID_NUMBERS)
def test_nonfinite_presets_neither_create_nor_overwrite(
    numeric_client, endpoint, run_type, field, number
):
    client, store, db = numeric_client
    finite = {
        "name": "owned",
        "run_type": run_type,
        "parameters": {
            "initial_cash": 1234.5,
            "transaction_cost_rate": 0,
            "risk_free_rate": 0,
        },
    }
    created = client.post("/presets", json=finite)
    assert created.status_code == 201
    saved = created.json()
    assert all(
        saved["parameters"][key] == value for key, value in finite["parameters"].items()
    )
    prefix = json.dumps({"name": "invalid", "run_type": run_type})[:-1]
    body = prefix + f',"parameters":{{"{field}":{number}}}}}'
    for method, path in [
        (client.post, "/presets"),
        (client.put, f"/presets/{saved['id']}"),
    ]:
        response = method(
            path, content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "finite_number"
        assert field in response.json()["detail"][0]["loc"]
        json.loads(response.text, parse_constant=lambda value: pytest.fail(value))
    assert db.query(PresetRow).count() == 1
    assert client.get("/presets", params={"run_type": run_type}).json() == [saved]
    store.create.assert_not_called()


def test_other_validation_errors_keep_fastapi_detail(numeric_client):
    client, store, _ = numeric_client
    for content in [
        '{"initial_cash":0}',
        '{"initial_cash":""}',
        '{"data_source":"invalid"}',
        "{",
    ]:
        response = client.post(
            "/runs", content=content, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][0] == "body"
    store.create.assert_not_called()
