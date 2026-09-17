#!/usr/bin/env python3
"""Generate or check the non-circular SparkPipe source-package manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from package_inventory import PACKAGE_MANIFEST_NAME, source_inventory


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / PACKAGE_MANIFEST_NAME


def build_manifest(root: Path) -> dict:
    entries = source_inventory(root)
    return {
        "manifest_schema": 2,
        "package_kind": "sparkpipe_source",
        "inventory_policy": "tools/package_inventory.py:v2",
        "file_count": len(entries),
        "total_bytes": sum(entry.size_bytes for entry in entries),
        "files": [
            {
                "path": entry.path,
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            }
            for entry in entries
        ],
    }


def render_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    output_path = arguments.output or (root / PACKAGE_MANIFEST_NAME)
    expected = render_manifest(build_manifest(root))
    if arguments.check:
        if not output_path.is_file():
            raise SystemExit(f"missing package manifest: {output_path}")
        actual = output_path.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit("PACKAGE_MANIFEST.json is stale")
        print("package manifest matches the source payload")
        return 0
    output_path.write_text(expected, encoding="utf-8")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
