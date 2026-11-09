"""The supported Linux targets must resolve CPU inference, including Intel Macs."""

from pathlib import Path
import tomllib

from packaging.markers import Marker
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "platform,machine,cpu",
    [
        ("linux", "x86_64", True),
        ("linux", "aarch64", True),
        ("darwin", "arm64", False),
        ("darwin", "x86_64", False),
        ("win32", "AMD64", False),
    ],
)
def test_torch_source_is_unambiguous_and_linux_is_cpu_only(platform, machine, cpu):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    candidates = project["tool"]["poetry"]["dependencies"]["torch"]
    selected = [
        entry
        for entry in candidates
        if Marker(entry["markers"]).evaluate(
            {"sys_platform": platform, "platform_machine": machine}
        )
    ]
    assert len(selected) == 1
    assert selected[0]["version"].endswith("+cpu") is cpu
    assert selected[0]["source"] == ("pytorch-cpu" if cpu else "pypi")
    lock = tomllib.loads((ROOT / "poetry.lock").read_text())
    package = next(
        record
        for record in lock["package"]
        if record["name"] == "torch" and record["version"] == selected[0]["version"]
    )
    if cpu:
        assert package["source"]["url"] == "https://download.pytorch.org/whl/cpu"
        wheels = [item["file"] for item in package["files"]]
        assert any("cp311" in wheel and machine in wheel for wheel in wheels)
        assert not any(
            name.startswith(("nvidia-", "cuda-", "triton"))
            for name in package.get("dependencies", {})
        )
