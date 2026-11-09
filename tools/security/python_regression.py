"""Reject newly matching package/advisory pairs between complete PyPI lock scans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from python_audit import collect_report, post_queries


def advisory_pairs(report: dict) -> set[tuple[str, str]]:
    if report.get("pagination_complete") is not True:
        raise ValueError("Incomplete inventory cannot establish a baseline")
    packages = report.get("packages")
    if (
        not isinstance(packages, list)
        or not packages
        or report.get("packages_queried") != len(packages)
    ):
        raise ValueError("Incomplete package inventory")
    pairs: set[tuple[str, str]] = set()
    for package in packages:
        if not isinstance(package, dict):
            raise ValueError("Invalid package record")
        name = package.get("name")
        identifiers = package.get("advisory_ids")
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(identifiers, list)
            or any(not isinstance(item, str) or not item for item in identifiers)
        ):
            raise ValueError("Invalid package advisory inventory")
        pairs.update((name, identifier) for identifier in identifiers)
    return pairs


def compare_reports(base: dict, head: dict) -> dict:
    before, after = advisory_pairs(base), advisory_pairs(head)
    introduced = [
        {"package": name, "advisory_id": identifier}
        for name, identifier in sorted(after - before)
    ]
    return {
        "schema_version": 1,
        "comparison_complete": True,
        "introduced": introduced,
        "base_report": base,
        "head_report": head,
        "scope": (
            "All newly matching package/advisory ID pairs fail, regardless of "
            "severity, optional groups or platform markers. Aliases are not "
            "deduplicated. Existing findings remain release blockers requiring "
            "triage; this comparison grants no exception or security clearance. "
            "Two live collections are not an atomic advisory-database snapshot."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-lock", type=Path, required=True)
    parser.add_argument("--head-lock", type=Path, default=Path("poetry.lock"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        base = collect_report(args.base_lock.read_bytes(), post_queries)
        head = collect_report(args.head_lock.read_bytes(), post_queries)
        report = compare_reports(base, head)
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "comparison_complete": False,
            "error_type": type(exc).__name__,
        }
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print(
            f"Python advisory comparison failed: {type(exc).__name__}", file=sys.stderr
        )
        return 1
    if report["comparison_complete"] is not True:
        print(
            "Python advisory comparison failed: incomplete collection", file=sys.stderr
        )
        return 1
    introduced = report["introduced"]
    print(
        json.dumps({"comparison_complete": True, "introduced_pairs": len(introduced)})
    )
    return 1 if introduced else 0


if __name__ == "__main__":
    raise SystemExit(main())
