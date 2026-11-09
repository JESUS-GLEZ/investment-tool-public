"""Select slow-test work and fail closed when evaluating its required check."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

INPUT_FILES = {".python-version", "pyproject.toml", "poetry.lock"}
INPUT_PREFIXES = ("src/", "tests/", "examples/", "tools/ci/", ".github/workflows/")


def needs_slow_tests(paths: list[str]) -> bool:
    return any(
        path in INPUT_FILES or path.startswith(INPUT_PREFIXES) or path.endswith(".py")
        for path in paths
    )


def select_slow_tests(event_name: str, event: dict, cwd: Path | None = None) -> bool:
    """Compare complete Git trees; any unavailable/ambiguous input runs the suite."""
    try:
        if event_name == "pull_request":
            base = event["pull_request"]["base"]["sha"]
            head = event["pull_request"]["head"]["sha"]
            separator = "..."
        elif event_name == "push":
            base, head = event["before"], event["after"]
            separator = ".."
        else:
            return True
        if not all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{40}", sha)
            for sha in (base, head)
        ) or "0" * 40 in (base, head):
            return True
        result = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                f"{base}{separator}{head}",
                "--",
            ],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        paths = result.stdout.decode("utf-8").split("\0")
        return needs_slow_tests(paths)
    except (KeyError, TypeError, ValueError, OSError, subprocess.CalledProcessError):
        print(
            "Unable to establish changed paths; requiring the slow suite.",
            file=sys.stderr,
        )
        return True


def gate_passes(selection_result: str, required: str, slow_result: str) -> bool:
    return selection_result == "success" and (
        (required == "true" and slow_result == "success")
        or (required == "false" and slow_result == "skipped")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--event-name", required=True)
    select.add_argument("--event-path", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--selection-result", required=True)
    check.add_argument("--required", required=True)
    check.add_argument("--slow-result", required=True)
    args = parser.parse_args()
    if args.command == "select":
        try:
            event = json.loads(args.event_path.read_text())
            required = select_slow_tests(args.event_name, event)
        except (OSError, ValueError):
            required = True
        print(f"required={str(required).lower()}")
    elif not gate_passes(args.selection_result, args.required, args.slow_result):
        parser.exit(
            1,
            "Slow-test gate failed: selection or required test run did not succeed.\n",
        )
    else:
        print("Slow-test gate passed.")


if __name__ == "__main__":
    main()
