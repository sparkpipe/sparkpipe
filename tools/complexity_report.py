#!/usr/bin/env python3
"""Cyclomatic-complexity scan over the production trees (W4 redundancy lane).

Counts CCN per function (lizard-compatible rule: 1 + occurrences of
if / else-if / case / for / while / catch / && / || / ternary ?) over the
same function extraction as tools/dup_report.py, and prints every function
above --threshold (default 25, the house slop gate) plus the distribution
summary the merge gate cites (mean vs the 7.33 external-audit baseline).

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


def function_ccn(path: Path) -> list[dict]:
    rows = []
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    for fn in extract_functions(path):
        body = "\n".join(strip_code(text)[fn.start:fn.end + 1])
        decisions = len(DECISION_RE.findall(body))
        rows.append({
            "file": str(path.relative_to(ROOT)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=HOTSPOT_THRESHOLD)
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    rows = scan()
    hotspots = [row for row in rows if row["ccn"] > arguments.threshold]
    mean = sum(row["ccn"] for row in rows) / max(len(rows), 1)
    print("# Cyclomatic scan (tools/complexity_report.py)")
    print()
    print(f"- functions: {len(rows)}; mean CCN {mean:.2f} (external-audit baseline 7.33, max 157)")
    print(f"- hotspots > {arguments.threshold}: {len(hotspots)}")
    print()
    print("| CCN | lines | function |")
    print("|-----|-------|----------|")
    for row in hotspots:
        print(f"| {row['ccn']} | {row['lines']} | `{row['file']}:{row['function']}:{row['start']}` |")
    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
