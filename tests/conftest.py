import os
import sys
from pathlib import Path

import pytest

# Use a non-interactive backend for matplotlib in tests
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure the local 'src' folder is importable for tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_git_from_commit_hook(monkeypatch):
    """Synthetic Git subprocesses must never inherit the caller's repository."""
    for variable in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ):
        monkeypatch.delenv(variable, raising=False)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Give explicitly marked slow tests a larger default timeout."""
    for item in items:
        if (
            item.get_closest_marker("slow") is not None
            and item.get_closest_marker("timeout") is None
        ):
            item.add_marker(pytest.mark.timeout(300))
