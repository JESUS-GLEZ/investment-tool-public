"""Exercise the documented CLI in-process so pytest-socket covers its execution."""

import hashlib
import json
import runpy
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.timeout(120)
def test_offline_example_cli_exports_valid_repeatable_results(tmp_path, monkeypatch):
    for name in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    outputs = [tmp_path / "first", tmp_path / "repeat"]
    for output in outputs:
        monkeypatch.setattr(
            sys,
            "argv",
            ["offline_optimization.py", "--output-dir", str(output)],
        )
        runpy.run_path(
            str(ROOT / "examples/offline_optimization.py"), run_name="__main__"
        )
        assert {path.name for path in output.iterdir()} == {
            "inputs.json",
            "manifest.json",
            "feasible_front.csv",
            "front.png",
        }
        with Image.open(output / "front.png") as image:
            assert image.format == "PNG"
            assert image.width > 0 and image.height > 0
            image.verify()

    first, repeat = outputs
    for filename in ("inputs.json", "manifest.json", "feasible_front.csv"):
        assert (first / filename).read_bytes() == (repeat / filename).read_bytes()

    inputs = json.loads((first / "inputs.json").read_text())
    manifest = json.loads((first / "manifest.json").read_text())
    assert inputs["kind"] == "synthetic_optimisation_inputs"
    assert (
        manifest["input_sha256"]
        == hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    )
    front = pd.read_csv(first / "feasible_front.csv")
    assert len(front) == manifest["feasible_candidates"] > 0
    weights = front[inputs["assets"]].to_numpy()
    assert np.isfinite(front.to_numpy()).all()
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10, rtol=0)
    mu, cov = np.array(inputs["expected_returns"]), np.array(inputs["covariance"])
    np.testing.assert_allclose(front.estimated_return, weights @ mu, atol=1e-10)
    np.testing.assert_allclose(
        front.estimated_volatility,
        np.sqrt(np.einsum("ij,jk,ik->i", weights, cov, weights)),
        atol=1e-10,
    )
    selected = front.iloc[
        (front.estimated_return / front.estimated_volatility).argmax()
    ]
    for column in front.columns:
        assert manifest["selected"][column] == pytest.approx(selected[column])


def test_standalone_direct_dependency_pins_match_application_lock():
    lock = tomllib.loads((ROOT / "poetry.lock").read_text())
    versions = {package["name"]: package["version"] for package in lock["package"]}
    for line in (ROOT / "examples/requirements.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==")
            assert versions[name] == version, f"Update the standalone pin for {name}"
