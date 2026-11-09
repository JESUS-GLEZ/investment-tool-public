"""Verify changed-path selection and reject incomplete/failed dependency states."""

import contextlib
import io
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from slow_test_gate import gate_passes, needs_slow_tests, select_slow_tests

SCRIPT = Path(__file__).with_name("slow_test_gate.py")


class SlowTestGateTests(unittest.TestCase):
    def test_relevant_inputs_require_slow_tests(self):
        for path in [
            "src/config.ini",
            "tests/test_a.py",
            "entrypoint/execute.py",
            "migrations/a.py",
            "examples/requirements.txt",
            "tools/ci/slow_test_gate.py",
            ".github/workflows/slow-tests.yml",
            "pyproject.toml",
            "poetry.lock",
            ".python-version",
        ]:
            with self.subTest(path=path):
                self.assertTrue(needs_slow_tests([path]))

    def test_docs_and_frontend_only_skip_expensive_work(self):
        self.assertFalse(
            needs_slow_tests(
                [
                    "README.md",
                    "docs/testing.md",
                    "frontend/src/main.tsx",
                    "frontend/package-lock.json",
                ]
            )
        )
        self.assertFalse(needs_slow_tests([]))

    def test_git_comparison_includes_deleted_renamed_and_unusual_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    ["git", *args], cwd=root, text=True
                ).strip()

            git("init", "-q")
            git("config", "user.name", "CI test")
            git("config", "user.email", "ci-test@example.invalid")
            (root / "src").mkdir()
            (root / "src/old name\nfile.py").write_text("# sample\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            (root / "src/old name\nfile.py").rename(root / "README.md")
            git("add", "-A")
            git("commit", "-qm", "rename to docs")
            head = git("rev-parse", "HEAD")
            for name, event in [
                ("push", {"before": base, "after": head}),
                (
                    "pull_request",
                    {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}},
                ),
            ]:
                with self.subTest(event=name):
                    self.assertTrue(select_slow_tests(name, event, root))
            (root / "README.md").write_text("# docs change\n")
            git("add", ".")
            git("commit", "-qm", "docs only")
            self.assertFalse(
                select_slow_tests(
                    "push", {"before": head, "after": git("rev-parse", "HEAD")}, root
                )
            )

    def test_uncertain_event_or_diff_always_requires_suite(self):
        for name, event in [
            ("workflow_dispatch", {}),
            ("unknown", {}),
            ("push", {}),
            ("push", {"before": "0" * 40, "after": "a" * 40}),
            ("push", {"before": "--help", "after": "a" * 40}),
            ("pull_request", {"pull_request": None}),
        ]:
            with self.subTest(name=name, event=event), contextlib.redirect_stderr(
                io.StringIO()
            ):
                self.assertTrue(select_slow_tests(name, event))
        with patch(
            "slow_test_gate.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "git"),
        ), contextlib.redirect_stderr(io.StringIO()):
            self.assertTrue(
                select_slow_tests("push", {"before": "a" * 40, "after": "b" * 40})
            )

    def test_gate_accepts_only_completed_selection_and_correct_slow_state(self):
        results = ["success", "failure", "cancelled", "skipped", ""]
        for selection, required, slow in itertools.product(
            results, ["true", "false", "", "unknown"], results
        ):
            expected = (selection, required, slow) in [
                ("success", "true", "success"),
                ("success", "false", "skipped"),
            ]
            with self.subTest(selection=selection, required=required, slow=slow):
                self.assertEqual(gate_passes(selection, required, slow), expected)

    def test_actual_gate_cli_exit_codes(self):
        for selection, required, slow, code in [
            ("success", "true", "success", 0),
            ("success", "false", "skipped", 0),
            ("success", "true", "failure", 1),
            ("success", "true", "skipped", 1),
            ("failure", "false", "skipped", 1),
            ("cancelled", "true", "cancelled", 1),
            ("success", "", "skipped", 1),
        ]:
            with self.subTest(selection=selection, required=required, slow=slow):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "check",
                        "--selection-result",
                        selection,
                        "--required",
                        required,
                        "--slow-result",
                        slow,
                    ],
                    capture_output=True,
                )
                self.assertEqual(result.returncode, code)

    def test_actual_selection_cli_unknown_or_unreadable_event(self):
        with tempfile.TemporaryDirectory() as directory:
            event = Path(directory) / "event.json"
            for content in [json.dumps({}), "invalid json"]:
                event.write_text(content)
                result = subprocess.check_output(
                    [
                        sys.executable,
                        str(SCRIPT),
                        "select",
                        "--event-name",
                        "push",
                        "--event-path",
                        str(event),
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                self.assertEqual(result, "required=true\n")
            event.unlink()
            result = subprocess.check_output(
                [
                    sys.executable,
                    str(SCRIPT),
                    "select",
                    "--event-name",
                    "push",
                    "--event-path",
                    str(event),
                ],
                text=True,
            )
            self.assertEqual(result, "required=true\n")


if __name__ == "__main__":
    unittest.main()
