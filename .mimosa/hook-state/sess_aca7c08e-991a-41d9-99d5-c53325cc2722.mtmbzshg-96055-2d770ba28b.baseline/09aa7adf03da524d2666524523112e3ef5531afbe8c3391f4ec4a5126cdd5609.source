#!/usr/bin/env python3
"""Generate or check SHA256SUMS for the source payload and its manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from package_inventory import (
    CHECKSUM_MANIFEST_NAME,
    PACKAGE_MANIFEST_NAME,
    sha256_file,
    source_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def render_checksums(root: Path) -> str:
    paths = [entry.path for entry in source_inventory(root)]
    paths.append(PACKAGE_MANIFEST_NAME)
    paths = sorted(set(paths))
    lines = []
    for relative_path in paths:
        absolute_path = root / relative_path
        if not absolute_path.is_file():
            raise ValueError(f"missing checksum input: {relative_path}")
        lines.append(f"{sha256_file(absolute_path)}  {relative_path}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    output_path = arguments.output or (root / CHECKSUM_MANIFEST_NAME)
    expected = render_checksums(root)
    if arguments.check:
        if not output_path.is_file():
            raise SystemExit(f"missing checksum manifest: {output_path}")
        if output_path.read_text(encoding="utf-8") != expected:
            raise SystemExit("SHA256SUMS is stale")
        print("SHA256SUMS matches the source payload and package manifest")
        return 0
    output_path.write_text(expected, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
