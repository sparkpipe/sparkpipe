#!/usr/bin/env python3
"""Port a family module tree to a new family by ordered text substitution.

Donor-assembly port helper: reads each donor file, applies the longest-first
replacement map, writes the destination tree. Prints per-file substitution
counts so a silent zero-substitution port cannot pass unnoticed.
"""

import argparse
from pathlib import Path


def parse_file_spec(spec: str) -> tuple[Path, Path]:
    donor, _, destination = spec.partition("->")
    if not destination:
        raise argparse.ArgumentTypeError(f"file spec needs 'donor->dest': {spec}")
    return Path(donor.strip()), Path(destination.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--map", action="append", required=True,
                        help="OLD=NEW text replacement; longest OLD wins")
    parser.add_argument("--file", action="append", required=True, type=parse_file_spec,
                        help="donor->dest path relative to --repo")
    arguments = parser.parse_args()
    replacements = []
    for entry in arguments.map:
        old, _, new = entry.partition("=")
        if not new:
            parser.error(f"--map needs OLD=NEW: {entry}")
        replacements.append((old, new))
    replacements.sort(key=lambda pair: -len(pair[0]))
    failures = 0
    for donor, destination in arguments.file:
        source = arguments.repo / donor
        target = arguments.repo / destination
        if not source.is_file():
            print(f"MISSING donor {source}")
            failures += 1
            continue
        text = source.read_text(encoding="utf-8")
        counts = []
        for old, new in replacements:
            found = text.count(old)
            if found:
                text = text.replace(old, new)
            counts.append(f"{old}->{new}:{found}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"{destination}  {' '.join(counts)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
