"""Offline regression controls using exact lock scans and CLI failure behavior."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_audit import TORCH_CPU_SOURCE, collect_report
from python_regression import compare_reports, main

BASE = b"""[[package]]
name = "some_package"
version = "1.0"
optional = true
groups = ["dev"]
[metadata]
lock-version = "2.0"
"""


def scan(lock, findings):
    return collect_report(
        lock,
        lambda queries: {
            "results": [
                {"vulns": [{"id": item} for item in findings.get(q["version"], [])]}
                for q in queries
            ]
        },
    )


class PythonRegressionTests(unittest.TestCase):
    def test_added_vulnerable_cpu_build_is_not_hidden_by_a_clean_pypi_record(self):
        base = b'[[package]]\nname="torch"\nversion="2.14.1"\n'
        metadata = b'[metadata]\nlock-version="2.0"\n'
        cpu = b'[[package]]\nname="torch"\nversion="2.9.0+cpu"\n[package.source]\n'
        cpu += "\n".join(f'{k}="{v}"' for k, v in TORCH_CPU_SOURCE.items()).encode()
        findings = {"2.9.0": ["GHSA-old-cpu-control"]}
        comparison = compare_reports(
            scan(base + metadata, findings),
            scan(base + cpu + b"\n" + metadata, findings),
        )
        self.assertEqual(
            comparison["introduced"],
            [{"package": "torch", "advisory_id": "GHSA-old-cpu-control"}],
        )
        self.assertEqual(comparison["head_report"]["packages_queried"], 2)

    def test_actual_cli_missing_baseline_fails_and_retains_error_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "report.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("python_regression.py")),
                    "--base-lock",
                    str(root / "missing.lock"),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(output.read_text())["comparison_complete"])
            self.assertNotIn("missing.lock", result.stderr)

    def test_unwritable_report_path_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "poetry.lock"
            lock.write_bytes(BASE)
            with patch(
                "python_regression.post_queries", return_value={"results": [{}]}
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--base-lock",
                            str(lock),
                            "--head-lock",
                            str(lock),
                            "--output",
                            str(root),
                        ]
                    ),
                    1,
                )

    def test_existing_findings_and_removed_advisories_do_not_regress(self):
        before = scan(BASE, {"1.0": ["GHSA-existing", "PYSEC-alias"]})
        for ids in [["GHSA-existing", "PYSEC-alias"], ["GHSA-existing"], []]:
            with self.subTest(ids=ids):
                result = compare_reports(before, scan(BASE, {"1.0": ids}))
                self.assertEqual(result["introduced"], [])
                self.assertEqual(result["base_report"], before)

    def test_changed_version_new_id_fails_even_if_old_id_is_removed(self):
        before = scan(BASE, {"1.0": ["GHSA-old"]})
        after = scan(BASE.replace(b'"1.0"', b'"2.0"'), {"2.0": ["GHSA-new"]})
        self.assertEqual(
            compare_reports(before, after)["introduced"],
            [{"package": "some-package", "advisory_id": "GHSA-new"}],
        )

    def test_package_identity_optional_records_and_multiple_versions_are_preserved(
        self,
    ):
        head_lock = BASE.replace(
            b"[metadata]",
            b'[[package]]\nname="other"\nversion="2.0"\noptional=true\n'
            b'[[package]]\nname="some-package"\nversion="3.0"\n[metadata]',
        )
        result = compare_reports(
            scan(BASE, {"1.0": ["GHSA-shared"]}),
            scan(
                head_lock,
                {
                    "1.0": ["GHSA-shared"],
                    "2.0": ["GHSA-shared"],
                    "3.0": ["GHSA-new", "GHSA-new"],
                },
            ),
        )
        self.assertEqual(
            result["introduced"],
            [
                {"package": "other", "advisory_id": "GHSA-shared"},
                {"package": "some-package", "advisory_id": "GHSA-new"},
            ],
        )

    def test_incomplete_or_invalid_inventory_never_becomes_clean_baseline(self):
        good = scan(BASE, {})
        for bad in [
            {},
            {**good, "pagination_complete": False},
            {**good, "packages_queried": 2},
            {**good, "packages": []},
            {**good, "packages": [None]},
            {**good, "packages": [{"name": "some", "advisory_ids": [None]}]},
        ]:
            for base, head in [(bad, good), (good, bad)]:
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    compare_reports(base, head)

    def test_cli_keeps_complete_reports_for_clean_and_introduced_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, head, output = (
                root / "base.lock",
                root / "head.lock",
                root / "report.json",
            )
            base.write_bytes(BASE)
            head.write_bytes(BASE.replace(b'"1.0"', b'"2.0"'))
            for ids, expected in [([], 0), (["GHSA-new"], 1)]:

                def post(queries):
                    return {
                        "results": [
                            {
                                "vulns": [
                                    {"id": item}
                                    for item in (ids if q["version"] == "2.0" else [])
                                ]
                            }
                            for q in queries
                        ]
                    }

                with patch(
                    "python_regression.post_queries", side_effect=post
                ), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        main(
                            [
                                "--base-lock",
                                str(base),
                                "--head-lock",
                                str(head),
                                "--output",
                                str(output),
                            ]
                        ),
                        expected,
                    )
                result = json.loads(output.read_text())
                self.assertTrue(result["comparison_complete"])
                self.assertNotEqual(
                    result["base_report"]["lock_sha256"],
                    result["head_report"]["lock_sha256"],
                )

    def test_cli_base_or_head_collection_failure_replaces_stale_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock, output = root / "poetry.lock", root / "report.json"
            lock.write_bytes(BASE)
            good = scan(BASE, {})
            for side_effect in [[OSError("offline")], [good, ValueError("incomplete")]]:
                output.write_text('{"comparison_complete":true,"introduced":[]}')
                with patch(
                    "python_regression.collect_report", side_effect=side_effect
                ), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(
                        main(
                            [
                                "--base-lock",
                                str(lock),
                                "--head-lock",
                                str(lock),
                                "--output",
                                str(output),
                            ]
                        ),
                        1,
                    )
                result = json.loads(output.read_text())
                self.assertFalse(result["comparison_complete"])
                self.assertNotIn("introduced", result)


if __name__ == "__main__":
    unittest.main()
