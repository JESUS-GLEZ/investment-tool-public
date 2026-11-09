"""Use owned files to exercise the dataset boundary before deserialization."""

import os
import pickle
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import adapters
from api.auth import get_current_user
from api.main import create_app
from api.models import UserRow
from api.routers import defaults
from api.store import RunStore
from investment_tool.exceptions import DataValidationError


@pytest.fixture()
def datasets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "input"
    root.mkdir()
    good = root / "trusted prices.pkl"
    good.write_bytes(
        pickle.dumps(
            {
                "stock_dict": {
                    "SAFE": pd.DataFrame(
                        index=pd.to_datetime(["2020-01-01", "2021-01-01"])
                    )
                }
            }
        )
    )
    outside = tmp_path / "outside.pkl"
    outside.write_bytes(pickle.dumps({"stock_dict": {"OUTSIDE": None}}))
    monkeypatch.setattr(adapters, "_INPUT_BASE", root)
    monkeypatch.setattr(defaults, "_INPUT_BASE", root)
    monkeypatch.setattr(defaults, "_CATALOG_BASE", tmp_path / "catalog")
    store = RunStore()
    app = create_app(store=store)
    app.dependency_overrides[get_current_user] = lambda: UserRow(
        id=1, email="owned@example.test", password_hash="unused"
    )
    return TestClient(app), root, good, outside, store


@pytest.mark.parametrize(
    "name",
    [
        "../outside.pkl",
        "nested/prices.pkl",
        r"..\outside.pkl",
        r"C:\outside.pkl",
        "C:outside.pkl",
        "",
        "safe.txt",
        "bad\x00.pkl",
    ],
)
def test_info_rejects_paths_before_pickle_load(datasets, name):
    client, _, _, _, _ = datasets
    with patch.object(defaults.pickle, "load") as load:
        response = client.get("/data/info", params={"file": name})
    assert response.status_code == 404
    load.assert_not_called()


def test_absolute_path_is_rejected_by_info_and_worker_adapter(datasets):
    client, _, _, outside, _ = datasets
    with patch.object(defaults.pickle, "load") as load:
        response = client.get("/data/info", params={"file": str(outside)})
    assert response.status_code == 404
    assert str(outside) not in response.text
    load.assert_not_called()
    with pytest.raises(DataValidationError):
        adapters._resolve_local_data_path(str(outside))


@pytest.mark.parametrize(
    "endpoint",
    [
        "/runs",
        "/comparisons/strategy",
        "/comparisons/solver",
        "/comparisons/prediction",
    ],
)
@pytest.mark.parametrize("name", ["../outside.pkl", r"C:\outside.pkl"])
def test_local_run_requests_reject_paths_before_creating_a_run(
    datasets, endpoint, name
):
    client, _, _, _, store = datasets
    response = client.post(endpoint, json={"data_source": "local", "input_file": name})
    assert response.status_code == 422
    assert store.list_all() == []


@pytest.mark.parametrize("kind", ["escape", "broken", "directory", "loop"])
def test_invalid_catalog_entries_are_never_loaded_or_selected_as_latest(datasets, kind):
    client, root, good, outside, _ = datasets
    invalid = root / "invalid.pkl"
    if kind == "escape":
        invalid.symlink_to(outside)
    elif kind == "broken":
        invalid.symlink_to(root / "missing.pkl")
    elif kind == "directory":
        invalid.mkdir()
    else:
        invalid.symlink_to(invalid)
    os.utime(good, (1, 1))
    with patch.object(defaults.pickle, "load", wraps=pickle.load) as load:
        response = client.get("/data/info", params={"file": invalid.name})
        assert response.status_code == 404
        load.assert_not_called()
        listed = client.get("/data/files")
        assert [entry["file"] for entry in listed.json()] == [good.name]
        latest = client.get("/data/info")
    assert latest.status_code == 200
    assert latest.json()["stocks"] == ["SAFE"]
    assert latest.json()["file"] == good.name
    assert load.call_count == 1
    with pytest.raises(DataValidationError):
        adapters._resolve_local_data_path(invalid.name)


def test_valid_file_and_internal_alias_preserve_metadata_and_selection_name(datasets):
    client, root, good, _, _ = datasets
    alias = root / "alias.pkl"
    alias.symlink_to(good)
    for name in [good.name, alias.name]:
        response = client.get("/data/info", params={"file": name})
        assert response.status_code == 200
        expected = {
            "file": name,
            "frequency": None,
            "asset_config_file": None,
            "asset_config_name": None,
            "asset_config_available": None,
            "stocks": ["SAFE"],
            "crypto": [],
            "start_date": "2020-01-01",
            "end_date": "2021-01-01",
            "stock_count": 1,
            "crypto_count": 0,
            "time_contract_version": None,
        }
        assert {key: response.json()[key] for key in expected} == expected
        assert response.json()["features"]["cached_sentiment"] is None
        assert adapters._resolve_local_data_path(name) == good.resolve()


def test_only_invalid_entries_leave_latest_empty(datasets):
    client, root, good, outside, _ = datasets
    good.unlink()
    (root / "escaped.pkl").symlink_to(outside)
    assert client.get("/data/files").json() == []
    with patch.object(defaults.pickle, "load") as load:
        assert client.get("/data/info").status_code == 404
    load.assert_not_called()


def test_load_errors_do_not_expose_server_paths_or_raw_exceptions(datasets):
    client, root, good, _, _ = datasets
    with patch.object(
        defaults.pickle, "load", side_effect=ValueError(f"private {root}")
    ):
        response = client.get("/data/info", params={"file": good.name})
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to load local dataset."
    assert str(root) not in response.text


@pytest.mark.parametrize("frequency", ["daily", "weekly", "monthly"])
def test_info_uses_recorded_frequency_and_universe(datasets, frequency):
    client, _, good, _, _ = datasets
    data = pickle.loads(good.read_bytes())
    data["metadata"] = {
        "frequency": frequency,
        "asset_config_file": "example_asset_universe.json",
    }
    good.write_bytes(pickle.dumps(data))
    info = client.get("/data/info", params={"file": good.name}).json()
    assert info["frequency"] == frequency
    assert info["asset_config_file"] == "example_asset_universe.json"
    assert info["asset_config_name"]
    assert info["asset_config_available"] is True


def test_info_reports_missing_recorded_universe_without_replacing_it(datasets):
    client, _, good, _, _ = datasets
    data = pickle.loads(good.read_bytes())
    data["metadata"] = {"asset_config_file": "missing-universe.json"}
    good.write_bytes(pickle.dumps(data))
    info = client.get("/data/info", params={"file": good.name}).json()
    assert info["asset_config_file"] == "missing-universe.json"
    assert info["asset_config_available"] is False
    assert info["asset_config_name"] is None


def test_info_recovers_frequency_from_custom_filename_without_loading_catalog(datasets):
    client, root, good, _, _ = datasets
    custom = root / "input_monthly_validation_2018_filtered_news_2016_2018.pkl"
    custom.write_bytes(good.read_bytes())
    with patch.object(defaults.pickle, "load", wraps=pickle.load) as load:
        listed = client.get("/data/files")
        assert listed.status_code == 200
        load.assert_not_called()
        info = client.get("/data/info", params={"file": custom.name}).json()
    assert info["frequency"] == "monthly"
    assert info["asset_config_file"] is None


def test_inspection_caches_metadata_without_changing_dataset_or_loading_catalog(
    datasets,
):
    client, _, good, _, _ = datasets
    before = good.read_bytes()
    info = client.get("/data/info", params={"file": good.name}).json()
    assert info["features"] == {
        "prices": True,
        "execution_quotes": False,
        "news": False,
        "cached_sentiment": None,
    }
    assert info["provider"] is None
    with patch.object(defaults.pickle, "load") as load:
        summary = client.get("/data/files").json()[0]
        load.assert_not_called()
    assert summary["title"] == info["title"]
    assert summary["stock_count"] == 1
    assert good.read_bytes() == before
    good.write_bytes(pickle.dumps({"stock_dict": {}}))
    assert "title" not in client.get("/data/files").json()[0]


def test_cache_failure_does_not_fail_inspection(datasets):
    client, _, good, _, _ = datasets
    with patch(
        "api.dataset_catalog.tempfile.NamedTemporaryFile", side_effect=PermissionError
    ):
        response = client.get("/data/info", params={"file": good.name})
    assert response.status_code == 200
    assert response.json()["stock_count"] == 1


@pytest.mark.parametrize("version", [{}, [], "1", True])
def test_info_does_not_expose_malformed_time_contract_to_ui(datasets, version):
    client, _, good, _, _ = datasets
    data = pickle.loads(good.read_bytes())
    data["metadata"] = {"time_contract_version": version}
    good.write_bytes(pickle.dumps(data))
    response = client.get("/data/info", params={"file": good.name})
    assert response.status_code == 200
    assert response.json()["time_contract_version"] is None


def test_inspected_unknown_coverage_replaces_requested_filename_dates(datasets):
    client, root, _, _, _ = datasets
    path = root / "input_data_monthly_2020-01-01_to_2021-12-31_9stocks_saved.pkl"
    path.write_bytes(pickle.dumps({"stock_dict": {}}))
    client.get("/data/info", params={"file": path.name})
    entry = next(
        item for item in client.get("/data/files").json() if item["file"] == path.name
    )
    assert entry["start_date"] is None
    assert entry["end_date"] is None
    assert entry["stock_count"] == 0
