"""Exercise scan completeness and failures without contacting the advisory service."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from python_audit import TORCH_CPU_SOURCE, collect_report, main, post_queries

LOCK = b"""[[package]]
name = "Some_Package"
version = "1.2.3"
optional = true
[[package]]
name = "other"
version = "2.0"
optional = false
[metadata]
lock-version = "2.0"
"""


class PythonAuditTests(unittest.TestCase):
    def cpu_lock(self, name="torch", version="2.9.0+cpu", source=None):
        source = TORCH_CPU_SOURCE if source is None else source
        lines = ["[[package]]", f'name="{name}"', f'version="{version}"']
        lines.append("[package.source]")
        lines.extend(f'{key}="{value}"' for key, value in source.items())
        lines.extend(["[metadata]", 'lock-version="2.0"'])
        return "\n".join(lines).encode()

    def test_official_cpu_build_cannot_hide_public_release_advisories(self):
        calls = []

        def post(queries):
            calls.extend(queries)
            self.assertEqual(queries[0]["version"], "2.9.0")
            return {"results": [{"vulns": [{"id": "GHSA-vulnerable-control"}]}]}

        report = collect_report(self.cpu_lock(), post)
        package = report["packages"][0]
        self.assertEqual(package["version"], "2.9.0+cpu")
        self.assertEqual(package["advisory_query_version"], "2.9.0")
        self.assertEqual(package["package_source"], TORCH_CPU_SOURCE)
        self.assertEqual(package["advisory_ids"], ["GHSA-vulnerable-control"])
        self.assertEqual(report["matched_package_names"], 1)
        self.assertEqual(len(calls), 1)

    def test_cpu_and_pypi_records_are_both_scanned_without_losing_identity(self):
        lock = self.cpu_lock().replace(
            b"[metadata]", b'[[package]]\nname="torch"\nversion="2.14.1"\n[metadata]'
        )

        def post(queries):
            self.assertEqual([q["version"] for q in queries], ["2.9.0", "2.14.1"])
            return {"results": [{"vulns": [{"id": "GHSA-old-cpu"}]}, {}]}

        report = collect_report(lock, post)
        self.assertEqual(report["packages_queried"], 2)
        self.assertEqual(
            [p["version"] for p in report["packages"]], ["2.9.0+cpu", "2.14.1"]
        )
        self.assertIsNone(report["packages"][1]["package_source"])

    def test_cpu_exception_rejects_other_packages_sources_and_build_variants(self):
        invalid = [self.cpu_lock(name="some-package")]
        for key, value in [
            ("url", "https://download.pytorch.org/whl/cpu/evil"),
            ("url", "https://download.pytorch.org.evil.test/whl/cpu"),
            ("url", "http://download.pytorch.org/whl/cpu"),
            ("url", "https://download.pytorch.org/whl/cpu?other=source"),
            ("reference", "other"),
            ("type", "git"),
        ]:
            invalid.append(self.cpu_lock(source={**TORCH_CPU_SOURCE, key: value}))
        for version in ["2.14.1", "2.14.1+cu130", "2.14.1+cpu.other", "2.14.1rc1+cpu"]:
            invalid.append(self.cpu_lock(version=version))
        for lock in invalid:
            with self.subTest(lock=lock), self.assertRaises(ValueError):
                collect_report(lock, lambda _: self.fail("Invalid source reached OSV"))

    def test_exact_versions_optional_records_and_alias_ids_remain_visible(self):
        calls = []

        def post(queries):
            calls.append(queries)
            return {
                "results": [
                    {
                        "vulns": [
                            {"id": "GHSA-test"},
                            {"id": "PYSEC-test"},
                            {"id": "GHSA-test"},
                        ]
                    },
                    {},
                ]
            }

        report = collect_report(LOCK, post)
        self.assertEqual(
            calls[0][0],
            {
                "package": {"ecosystem": "PyPI", "name": "some-package"},
                "version": "1.2.3",
            },
        )
        self.assertEqual(report["packages_queried"], 2)
        self.assertEqual(report["matched_package_names"], 1)
        self.assertEqual(report["lock_sha256"], hashlib.sha256(LOCK).hexdigest())
        self.assertTrue(report["packages"][0]["optional"])
        self.assertEqual(
            report["packages"][0]["advisory_ids"], ["GHSA-test", "PYSEC-test"]
        )
        self.assertTrue(report["pagination_complete"])

    def test_only_paginated_queries_continue_and_results_keep_their_package(self):
        calls = []

        def post(queries):
            calls.append(queries)
            if len(calls) == 1:
                return {
                    "results": [
                        {},
                        {"vulns": [{"id": "GHSA-first"}], "next_page_token": "next"},
                    ]
                }
            self.assertEqual(len(queries), 1)
            self.assertEqual(queries[0]["package"]["name"], "other")
            self.assertEqual(queries[0]["page_token"], "next")
            return {"results": [{"vulns": [{"id": "GHSA-last"}]}]}

        report = collect_report(LOCK, post)
        self.assertEqual(report["packages"][0]["advisory_ids"], [])
        self.assertEqual(
            report["packages"][1]["advisory_ids"], ["GHSA-first", "GHSA-last"]
        )

    def test_batches_include_every_locked_record_and_multiple_versions(self):
        lock = (
            b"\n".join(
                b'[[package]]\nname="same"\nversion="1.' + str(i).encode() + b'"'
                for i in range(205)
            )
            + b'\n[metadata]\nlock-version="2.1"\n'
        )
        sizes = []

        def post(queries):
            sizes.append(len(queries))
            return {"results": [{} for _ in queries]}

        report = collect_report(lock, post)
        self.assertEqual(sizes, [100, 100, 5])
        self.assertEqual(report["packages_queried"], 205)
        self.assertEqual(report["matched_package_names"], 0)

    def test_rejects_missing_records_unknown_formats_and_custom_sources(self):
        for raw in [
            b"",
            b'[metadata]\nlock-version="2.0"',
            LOCK.replace(b'"2.0"\n', b'"3.0"\n'),
            LOCK.replace(b"optional = true", b'[package.source]\ntype="git"'),
            LOCK.replace(b"Some_Package", b"bad/name"),
        ]:
            with self.subTest(lock=raw), self.assertRaises(ValueError):
                collect_report(raw, lambda _: self.fail("Invalid input reached OSV"))

    def test_rejects_incomplete_or_malformed_service_responses(self):
        for payload in [
            None,
            {},
            {"results": []},
            {"results": [{}, {}, {}]},
            {"results": [{"vulns": None}, {}]},
            {"results": [{"vulns": [{}]}, {}]},
            {"results": [{"error": "failed"}, {}]},
            {"results": [{"next_page_token": 1}, {}]},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                collect_report(LOCK, lambda _: payload)

    def test_repeated_or_unbounded_pagination_fails(self):
        with self.assertRaises(ValueError):
            collect_report(
                LOCK, lambda qs: {"results": [{"next_page_token": "same"} for _ in qs]}
            )
        count = 0

        def endless(queries):
            nonlocal count
            count += 1
            return {"results": [{"next_page_token": str(count)} for _ in queries]}

        with self.assertRaises(ValueError):
            collect_report(LOCK, endless)
        self.assertEqual(count, 10)

    def test_transport_sends_only_json_queries_to_the_fixed_https_endpoint(self):
        response = io.BytesIO(b'{"results": [{}]}')
        queries = [
            {"package": {"ecosystem": "PyPI", "name": "example"}, "version": "1.0"}
        ]
        with patch(
            "python_audit.urllib.request.urlopen", return_value=response
        ) as send:
            self.assertEqual(post_queries(queries), {"results": [{}]})
        request = send.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.osv.dev/v1/querybatch")
        self.assertEqual(request.method, "POST")
        self.assertEqual(json.loads(request.data), {"queries": queries})
        self.assertEqual(send.call_args.kwargs, {"timeout": 30})

    def test_cli_reports_findings_but_rejects_failed_collection_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            lock, output = (
                Path(directory) / "poetry.lock",
                Path(directory) / "result.json",
            )
            lock.write_bytes(LOCK)
            args = ["--lock", str(lock), "--output", str(output)]
            with patch(
                "python_audit.post_queries",
                return_value={"results": [{"vulns": [{"id": "GHSA-test"}]}, {}]},
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(json.loads(output.read_text())["matched_package_names"], 1)
            output.unlink()
            with patch(
                "python_audit.post_queries", side_effect=OSError("offline")
            ), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
