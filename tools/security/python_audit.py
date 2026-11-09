"""Scan every locked PyPI record and the narrowly allowed official Torch CPU build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

ENDPOINT = "https://api.osv.dev/v1/querybatch"
BATCH_SIZE = 100
MAX_PAGES = 10
TORCH_CPU_SOURCE = {
    "type": "legacy",
    "url": "https://download.pytorch.org/whl/cpu",
    "reference": "pytorch-cpu",
}


def post_queries(queries: list[dict]) -> dict:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps({"queries": queries}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError("Invalid OSV batch response")
    return payload


def collect_report(
    lock_bytes: bytes, post: Callable[[list[dict]], dict] = post_queries
) -> dict:
    lock = tomllib.loads(lock_bytes.decode("utf-8"))
    metadata = lock.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("lock-version") not in {
        "2.0",
        "2.1",
    }:
        raise ValueError("Unsupported Poetry lock format")
    records = lock.get("package")
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty Poetry package inventory")
    packages = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Invalid locked package record")
        name, version = record.get("name"), record.get("version")
        if (
            not isinstance(name, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
            or not isinstance(version, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]*", version) is None
        ):
            raise ValueError("Invalid locked package name or exact version")
        name = re.sub(r"[-_.]+", "-", name).lower()
        query_version = version
        source = record.get("source")
        if source is not None:
            cpu_version = re.fullmatch(r"([0-9]+\.[0-9]+\.[0-9]+)\+cpu", version)
            if name != "torch" or source != TORCH_CPU_SOURCE or cpu_version is None:
                raise ValueError("Unsupported non-PyPI source or build variant")
            # OSV's PyPI ranges describe the public release. Preserve the exact
            # local build identity in the report, but do not let +cpu hide CVEs.
            query_version = cpu_version[1]
        packages.append(
            {
                "name": name,
                "version": version,
                "advisory_query_version": query_version,
                "package_source": source,
                "optional": record.get("optional"),
                "groups": record.get("groups"),
                "markers": record.get("markers"),
            }
        )
    queries = [
        {
            "package": {"ecosystem": "PyPI", "name": p["name"]},
            "version": p["advisory_query_version"],
        }
        for p in packages
    ]
    found: list[set[str]] = [set() for _ in queries]
    pending = list(enumerate(queries))
    seen_tokens: set[tuple[int, str]] = set()
    for _ in range(MAX_PAGES):
        if not pending:
            break
        next_pending = []
        for start in range(0, len(pending), BATCH_SIZE):
            batch = pending[start : start + BATCH_SIZE]
            payload = post([q for _, q in batch])
            if not isinstance(payload, dict) or "error" in payload:
                raise ValueError("Invalid OSV batch response")
            results = payload.get("results")
            if not isinstance(results, list) or len(results) != len(batch):
                raise ValueError("Incomplete OSV batch response")
            for (index, query), result in zip(batch, results):
                if not isinstance(result, dict) or "error" in result:
                    raise ValueError("Invalid OSV package result")
                vulns = result.get("vulns", [])
                if not isinstance(vulns, list):
                    raise ValueError("Invalid OSV advisory inventory")
                for vuln in vulns:
                    identifier = vuln.get("id") if isinstance(vuln, dict) else None
                    if (
                        not isinstance(identifier, str)
                        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", identifier)
                        is None
                    ):
                        raise ValueError("Invalid OSV advisory identifier")
                    found[index].add(identifier)
                token = result.get("next_page_token")
                if token is None or token == "":
                    continue
                if not isinstance(token, str) or (index, token) in seen_tokens:
                    raise ValueError("Invalid or repeated OSV pagination token")
                seen_tokens.add((index, token))
                next_pending.append((index, {**query, "page_token": token}))
        pending = next_pending
    if pending:
        raise ValueError("OSV pagination did not complete within its bounded limit")
    for package, identifiers in zip(packages, found):
        package["advisory_ids"] = sorted(identifiers)
    return {
        "schema_version": 1,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "source": ENDPOINT,
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "lock_format": lock["metadata"]["lock-version"],
        "packages_queried": len(packages),
        "matched_package_names": len(
            {p["name"] for p in packages if p["advisory_ids"]}
        ),
        "pagination_complete": True,
        "packages": packages,
        "scope": (
            "All locked PyPI records and official PyTorch CPU records, including "
            "optional/groups/platform variants. Only the exact pytorch-cpu "
            "source URL and torch X.Y.Z+cpu builds are allowed; their public "
            "X.Y.Z release is queried while exact build/source identity is retained. "
            "No applicability filtering, severity decision, alias/CVE deduplication, "
            "installed-environment, container, Actions or embedded-software audit. "
            "Successful collection does not establish release clearance."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path("poetry.lock"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = collect_report(args.lock.read_bytes(), post_queries)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(
            f"Python dependency inventory failed: {type(exc).__name__}", file=sys.stderr
        )
        return 1
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "packages_queried",
                    "matched_package_names",
                    "pagination_complete",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
