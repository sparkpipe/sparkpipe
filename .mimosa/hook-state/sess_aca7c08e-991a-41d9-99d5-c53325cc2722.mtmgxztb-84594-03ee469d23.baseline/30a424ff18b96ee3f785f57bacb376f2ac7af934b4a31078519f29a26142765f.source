#!/usr/bin/env python3
"""Cyclomatic-complexity scan over the production trees (W4 redundancy lane).

Counts CCN per function (lizard-compatible rule: 1 + occurrences of
if / else-if / case / for / while / catch / && / || / ternary ?) over the
same function extraction as tools/dup_report.py, and prints every function
above --threshold (default 25, the house slop gate) plus the distribution
summary the merge gate cites.

SCOPING (complexity lane, 2026-08-28 — the audit's own recommendation):
the merge-gate metric is the PRODUCTION scope only. Functions under any
`*/validation/*` tree (the per-family CUDA validation harnesses —
deliberately independent control-vs-candidate units per
docs/CLEANUP_PROGRAM.md) are reported in their own scope with their own
budget and never inflate the production numbers. The pre-split mixed
metric averaged 8.01 over 3275 functions with 5-8 of the top-25 being
validation-harness functions; the honest production-only baseline this
lane committed is mean 7.90 over 2945 functions (see
docs/AGENT_LANE_BRIEFS/reports/ccn-2026-08-28.md for the raw movement).

Usage:
    python3 tools/complexity_report.py               # hotspots > 25 on stdout
    python3 tools/complexity_report.py --json OUT    # all functions, machine-readable
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dup_report import ROOT, SCAN_ROOTS, EXTENSIONS, extract_functions, strip_code  # noqa: E402

DECISION_RE = re.compile(r"\b(?:if|case|for|while|catch)\b|&&|\|\||\?")
HOTSPOT_THRESHOLD = 25

# The external-audit baseline (mean 7.33) was a production-code baseline;
# the mixed scan compared it against a number inflated by validation
# harnesses. The production scope is the honest comparator.
BASELINE_MEAN = 7.33


def is_validation(path_relative: str) -> bool:
    """True when the file lives under a `validation/` harness tree.

    These are the per-family CUDA validation units (control-vs-candidate
    doctrine: deliberately independent, never merged, never shared).
    They are exercised by qualification, not by serving, so they carry
    their own complexity budget instead of inflating the production
    merge-gate metric.
    """
    return "validation" in Path(path_relative).parts


def function_ccn(path: Path) -> list[dict]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    for fn in extract_functions(path):
        body = "\n".join(strip_code(text)[fn.start:fn.end + 1])
        decisions = len(DECISION_RE.findall(body))
        relative = str(path.relative_to(ROOT))
        rows.append({
            "file": relative,
            "scope": "validation" if is_validation(relative) else "production",
            "function": fn.name,
            "start": fn.start + 1,
            "end": fn.end + 1,
            "lines": fn.end - fn.start + 1,
            "ccn": decisions + 1,
        })
    return rows


def scan() -> list[dict]:
    rows: list[dict] = []
    for root in SCAN_ROOTS:
        base = ROOT / root
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in EXTENSIONS and path.is_file():
                rows.extend(function_ccn(path))
    rows.sort(key=lambda row: -row["ccn"])
    return rows


def scope_rows(rows: list[dict], scope: str) -> list[dict]:
    return [row for row in rows if row["scope"] == scope]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=HOTSPOT_THRESHOLD)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    rows = scan()
    production = scope_rows(rows, "production")
    validation = scope_rows(rows, "validation")

    production_mean = sum(row["ccn"] for row in production) / max(len(production), 1)
    production_max = max((row["ccn"] for row in production), default=0)
    validation_mean = sum(row["ccn"] for row in validation) / max(len(validation), 1)
    validation_max = max((row["ccn"] for row in validation), default=0)

    hotspots = [row for row in production if row["ccn"] > arguments.threshold]
    validation_hotspots = [row for row in validation if row["ccn"] > arguments.threshold]

    print("# Cyclomatic scan (tools/complexity_report.py)")
    print()
    print("## Production scope (the merge-gate metric)")
    print()
    print(f"- functions: {len(production)}; mean CCN {production_mean:.2f} "
          f"(external-audit baseline {BASELINE_MEAN:.2f}); max CCN {production_max} "
          f"(committed ceiling: tests/test_complexity_ceiling.py)")
    print(f"- hotspots > {arguments.threshold}: {len(hotspots)}")
    print()
    print("| CCN | lines | function |")
    print("|-----|-------|----------|")
    for row in hotspots:
        print(f"| {row['ccn']} | {row['lines']} | `{row['file']}:{row['function']}:{row['start']}` |")
    if validation:
        print()
        print("## Validation scope (own budget — never in the production metric)")
        print()
        print(f"- functions: {len(validation)}; mean CCN {validation_mean:.2f}; "
              f"max CCN {validation_max}")
        print(f"- hotspots > {arguments.threshold}: {len(validation_hotspots)} "
              "(reported for visibility only; the validation harnesses are "
              "control-vs-candidate units by doctrine and are not merged into "
              "production, so they are budgeted separately)")
        print()
        print("| CCN | lines | function |")
        print("|-----|-------|----------|")
        for row in validation_hotspots:
            print(f"| {row['ccn']} | {row['lines']} | `{row['file']}:{row['function']}:{row['start']}` |")
    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
