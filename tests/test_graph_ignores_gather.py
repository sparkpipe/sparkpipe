#!/usr/bin/env python3
"""A model family whose stage geometry sets graph_ignores_gather never reads a step's gather fields.

The stage runner replays such a model's captured decode graph across steps whose gather_context and gather_sequence
differ (a prompt's last token, later decode steps, any prompt length), so a read of either field in that family's
sources would replay a graph built for another prompt.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARES = re.compile(r"graph_ignores_gather\s*=(?!\s*0u?\s*;)[^;]+;")
READS = re.compile(r"\bgather_(?:context|sequence)\b")
SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp"}


def family_sources(family):
    return sorted(path for path in family.rglob("*") if path.is_file() and path.suffix in SUFFIXES)


def main():
    failures, declaring = [], []
    for family in sorted(path for path in (ROOT / "model-families").iterdir() if path.is_dir()):
        sources = family_sources(family)
        if not any(DECLARES.search(path.read_text(errors="replace")) for path in sources):
            continue
        declaring.append(family.name)
        for path in sources:
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if READS.search(line):
                    failures.append(f"{path.relative_to(ROOT)}:{number}: reads a gather field but its family sets graph_ignores_gather")
    for failure in failures:
        print(f"FAIL {failure}")
    if failures:
        return 1
    print(f"PASS graph_ignores_gather: {len(declaring)} families declare it ({', '.join(declaring) or 'none'}) and none reads a gather field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
