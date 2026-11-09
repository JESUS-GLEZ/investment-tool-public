"""Asset manifests are the source of stock and crypto symbols."""

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from api.adapters import build_exec_params
from api.schemas import ExecutionRequest
from investment_tool.data.universes.loader import load_asset_universe
from investment_tool.config import AppConfig
from investment_tool.constants import DataSource, ExecutionMode, Frequency
from investment_tool.data.assembly.model_data_factory import ModelDataFactory
from investment_tool.data.repositories.base_repository import DataResult
from investment_tool.data.repositories.local_repository import LocalRepository


def test_example_manifest_is_default_and_includes_both_asset_classes() -> None:
    example = load_asset_universe()
    assert example.file == "example_asset_universe.json"
    assert set(example.stocks) == {"SPY", "QQQ", "IEF", "GLD", "AAPL", "MSFT"}
    assert example.crypto == ["BTC/USD"]


def test_saved_request_loads_archived_manifest_without_listing_it(
    tmp_path, monkeypatch
) -> None:
    from api.schemas import StrategyComparisonRequest
    from investment_tool.data.universes import loader

    monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
    manifest = {
        "name": "Test assets",
        "groups": {"stocks": ["GLD", "SPY"]},
        "crypto_symbols": [],
    }
    (tmp_path / "active.json").write_text(json.dumps(manifest))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "archived.json").write_text(
        json.dumps({**manifest, "crypto_symbols": ["BTC/USD"]})
    )
    request = StrategyComparisonRequest(
        asset_config_file="archived.json", strategy_types=["optimization"]
    )
    assert {u.file for u in loader.list_asset_universes()} == {"active.json"}
    universe = load_asset_universe(request.asset_config_file)
    assert universe.file == "archived.json"
    assert set(universe.stocks) == {"GLD", "SPY"}
    assert universe.crypto == ["BTC/USD"]


@pytest.mark.parametrize("outside_directory", [False, True])
def test_archive_symlinks_cannot_escape_config(
    tmp_path, monkeypatch, outside_directory: bool
) -> None:
    from investment_tool.data.universes import loader

    config = tmp_path / "config"
    config.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "escape.json").write_text(
        json.dumps(
            {"name": "Outside", "groups": {"stocks": ["AAPL"]}, "crypto_symbols": []}
        )
    )
    archive = config / "archive"
    if outside_directory:
        archive.symlink_to(outside, target_is_directory=True)
    else:
        archive.mkdir()
        (archive / "escape.json").symlink_to(outside / "escape.json")
    monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", config)
    with pytest.raises(ValueError, match="stay inside"):
        load_asset_universe("escape.json")


def test_factory_uses_selected_manifest_for_both_asset_types() -> None:
    factory = ModelDataFactory(
        DataSource.ALPACA,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        cfg=AppConfig(asset_config_file="example_asset_universe.json"),
    )
    stocks, crypto = factory._get_configured_symbols()
    assert len(stocks) == 6
    assert crypto == ["BTC/USD"]


def test_run_request_passes_selected_file_to_worker_configuration(
    tmp_path, monkeypatch
) -> None:
    from api import adapters

    monkeypatch.setattr(adapters, "_OUTPUT_BASE", tmp_path)
    request = ExecutionRequest(asset_config_file="example_asset_universe.json")
    parameters = build_exec_params(request, "asset-universe-test")
    assert parameters.cfg is not None
    assert parameters.cfg.asset_config_file == "example_asset_universe.json"


def test_run_request_rejects_unknown_manifest() -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        ExecutionRequest(asset_config_file="missing.json")


@pytest.mark.parametrize(
    "filename", ["../asset_universe.json", "/tmp/asset_universe.json", "missing.json"]
)
def test_invalid_manifest_name_is_rejected(filename: str) -> None:
    with pytest.raises(ValueError):
        load_asset_universe(filename)


def test_malformed_crypto_pair_is_rejected(tmp_path, monkeypatch) -> None:
    from investment_tool.data.universes import loader as asset_universe

    (tmp_path / "bad.json").write_text(
        json.dumps(
            {
                "name": "Bad",
                "groups": {"stocks": ["AAPL"]},
                "crypto_symbols": ["BTC"],
            }
        )
    )
    monkeypatch.setattr(asset_universe, "ASSET_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        load_asset_universe("bad.json")


def test_empty_crypto_list_excludes_local_crypto(monkeypatch) -> None:
    repository = LocalRepository(Frequency.MONTHLY, Path("unused.pkl"))
    prices = pd.DataFrame({"close": [1.0]})
    monkeypatch.setattr(
        repository,
        "_load_data",
        lambda: DataResult(
            stock_dict={"AAPL": prices}, crypto_dict={"BTC/USD": prices}
        ),
    )
    result = repository.fetch_all(["AAPL"], [], "2024-01-01", "2024-02-01")
    assert list(result.stock_dict) == ["AAPL"]
    assert result.crypto_dict == {}


def test_valuation_only_successors_are_fetched_but_not_candidates(
    tmp_path, monkeypatch
):
    from investment_tool.data.universes import loader

    manifest = {
        "name": "Merger",
        "groups": {"stocks": ["OLD"]},
        "crypto_symbols": [],
        "valuation_symbols": ["NEW"],
        "security_actions": [
            {
                "symbol": "OLD",
                "effective_at": "2020-04-03T12:00:00Z",
                "published_at": "2020-04-03T12:00:00Z",
                "successor_shares": {"NEW": 2},
                "source": "reviewed notice",
            }
        ],
    }
    (tmp_path / "merger.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
    universe = loader.load_asset_universe("merger.json")
    assert universe.stocks == ["OLD"]
    assert universe.quote_stocks == ["OLD", "NEW"]
    assert universe.summary()["stock_count"] == 1
    factory = ModelDataFactory(
        DataSource.LOCAL,
        ExecutionMode.BACKTESTING,
        Frequency.MONTHLY,
        0,
        input_data_path=Path("unused.pkl"),
        cfg=AppConfig(asset_config_file="merger.json"),
    )
    assert factory._get_configured_symbols() == (["OLD", "NEW"], [])


@pytest.mark.parametrize(
    "replacement", [{"NEW": 0}, {"NEW": float("nan")}, {"UNKNOWN": 2}, {"OLD": 2}]
)
def test_security_action_invalid_share_terms_are_rejected(
    tmp_path, monkeypatch, replacement
):
    from investment_tool.data.universes import loader

    manifest = {
        "name": "Bad",
        "groups": {"stocks": ["OLD"]},
        "crypto_symbols": [],
        "valuation_symbols": ["NEW"],
        "security_actions": [
            {
                "symbol": "OLD",
                "source": "notice",
                "successor_shares": replacement,
                "effective_at": "2020-04-03T12:00:00Z",
                "published_at": "2020-04-03T12:00:00Z",
            }
        ],
    }
    (tmp_path / "bad.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(loader, "ASSET_CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError):
        loader.load_asset_universe("bad.json")


def test_explicit_backtest_start_is_preserved_by_request_adapter(tmp_path, monkeypatch):
    from api import adapters

    monkeypatch.setattr(adapters, "_OUTPUT_BASE", tmp_path)
    request = ExecutionRequest(
        historical_start_date="2018-11-01",
        historical_end_date="2020-12-31",
        backtest_start_date="2020-01-01",
    )
    assert (
        build_exec_params(request, "annual-start").cfg.model.backtest_start_date
        == "2020-01-01"
    )


@pytest.mark.parametrize("start", ["2018-01-01", "2021-01-01", "invalid"])
def test_request_rejects_backtest_start_outside_history_or_invalid(start):
    with pytest.raises(ValidationError):
        ExecutionRequest(
            historical_start_date="2018-11-01",
            historical_end_date="2020-12-31",
            backtest_start_date=start,
        )
