"""Verify JSON precedence through the real four CLI parsers without execution."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from investment_tool.defaults import CONSTRUCTION, STRATEGY_SELECTION

ENTRYPOINTS = [
    "execute",
    "execute_strategy_comparison",
    "execute_solver_comparison",
    "execute_prediction_comparison",
]


@pytest.fixture(params=ENTRYPOINTS)
def cli(request, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "entrypoint"))
    return importlib.import_module(request.param)


def parse(cli, monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", [cli.__name__, *arguments])
    return cli.parse_arguments()


def write_params(tmp_path, values):
    path = tmp_path / "params.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return str(path)


def test_no_arguments_keep_shared_defaults(cli, monkeypatch):
    args = parse(cli, monkeypatch)
    assert args.data_source == STRATEGY_SELECTION.data_source
    assert args.frequency == STRATEGY_SELECTION.frequency
    assert args.save_input_data is False


def test_omitted_flags_preserve_json_values(cli, monkeypatch, tmp_path):
    path = write_params(
        tmp_path,
        {
            "data-source": "local",
            "frequency": "daily",
            "input-path": "synthetic.pkl",
            "output-dir": "example-output",
            "save-input-data": True,
            "validation-level": "strict",
        },
    )
    args = parse(cli, monkeypatch, "--use-json", path)
    assert args.data_source == "local"
    assert args.frequency == "daily"
    assert args.input_path == "synthetic.pkl"
    assert args.output_dir == "example-output"
    assert args.save_input_data is True
    assert args.validation_level == "strict"


def test_explicit_flags_override_json_even_when_equal_to_defaults(
    cli, monkeypatch, tmp_path
):
    path = write_params(
        tmp_path,
        {"data_source": "local", "frequency": "daily", "validation_level": "strict"},
    )
    args = parse(
        cli,
        monkeypatch,
        "--use-json",
        path,
        "--frequency=" + STRATEGY_SELECTION.frequency,
        "--data-source",
        STRATEGY_SELECTION.data_source,
        "--validation-level",
        "warn",
    )
    assert args.frequency == STRATEGY_SELECTION.frequency
    assert args.data_source == STRATEGY_SELECTION.data_source
    assert args.validation_level == "warn"


def test_sparse_json_keeps_defaults_for_missing_fields(cli, monkeypatch, tmp_path):
    path = write_params(tmp_path, {"input-path": "synthetic.pkl"})
    args = parse(cli, monkeypatch, "--use-json", path)
    assert args.frequency == STRATEGY_SELECTION.frequency
    assert args.data_source == STRATEGY_SELECTION.data_source
    assert args.output_dir is None
    assert args.save_input_data is False
    assert args.input_path == "synthetic.pkl"


def test_invalid_explicit_choice_still_fails(cli, monkeypatch, tmp_path):
    path = write_params(tmp_path, {"frequency": "daily"})
    with pytest.raises(SystemExit) as error:
        parse(cli, monkeypatch, "--use-json", path, "--frequency", "invalid")
    assert error.value.code == 2


def test_malformed_json_is_reported(cli, monkeypatch, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        parse(cli, monkeypatch, "--use-json", str(path))


@pytest.mark.parametrize(
    "module,flag,key",
    [
        (
            "execute_solver_comparison",
            "--no-markowitz-reference",
            "include_markowitz_reference",
        ),
        ("execute_prediction_comparison", "--no-sentiment", "compute_sentiment"),
    ],
)
def test_explicit_negative_switch_overrides_json(
    module, flag, key, monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "entrypoint"))
    cli = importlib.import_module(module)
    path = write_params(tmp_path, {key: True})
    assert getattr(parse(cli, monkeypatch, "--use-json", path, flag), key) is False


def test_strategy_comparison_explicit_values_win(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "entrypoint"))
    cli = importlib.import_module("execute_strategy_comparison")
    path = write_params(tmp_path, {"benchmark-asset": "SYNTHETIC", "min-assets": 9})
    args = parse(
        cli,
        monkeypatch,
        "--use-json",
        path,
        "--benchmark-asset",
        "SPY",
        "--min-assets",
        str(CONSTRUCTION.min_assets),
    )
    assert args.benchmark_asset == "SPY"
    assert args.min_assets == CONSTRUCTION.min_assets
