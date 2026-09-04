#!/usr/bin/env python3
"""Verify a SparkPipe source package without requiring a Git checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

from package_inventory import (
    CHECKSUM_MANIFEST_NAME,
    PACKAGE_MANIFEST_NAME,
    find_forbidden_packaged_paths,
    source_inventory,
)


ROOT = Path(__file__).resolve().parents[1]


def reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_manifest_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("manifest path must be a nonempty string")
    pure = PurePosixPath(value)
    if value.startswith("/") or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe manifest path: {value!r}")
    return value


def parse_checksums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line:
            continue
        pieces = raw_line.split("  ", 1)
        if len(pieces) != 2 or len(pieces[0]) != 64:
            raise ValueError(f"malformed SHA256SUMS line {line_number}")
        digest, relative_path = pieces
        if any(character not in "0123456789abcdef" for character in digest):
            raise ValueError(f"invalid digest on SHA256SUMS line {line_number}")
        relative_path = safe_manifest_path(relative_path)
        if relative_path in entries:
            raise ValueError(f"duplicate SHA256SUMS path: {relative_path}")
        entries[relative_path] = digest
    return entries


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--strict-package", action="store_true")
    arguments = parser.parse_args()

    root = arguments.root.resolve()
    manifest_path = root / PACKAGE_MANIFEST_NAME
    checksum_path = root / CHECKSUM_MANIFEST_NAME
    failures: list[str] = []

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_json_keys,
        )
    except Exception as error:
        raise SystemExit(f"cannot read package manifest: {error}") from error

    if manifest.get("manifest_schema") != 2:
        failures.append("manifest_schema must be 2")
    if manifest.get("package_kind") != "sparkpipe_source":
        failures.append("package_kind must be sparkpipe_source")
    if manifest.get("inventory_policy") != "tools/package_inventory.py:v2":
        failures.append("inventory_policy must be tools/package_inventory.py:v2")
    files = manifest.get("files")
    if not isinstance(files, list):
        failures.append("files must be a list")
        files = []

    recorded: dict[str, tuple[int, str]] = {}
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            failures.append(f"files[{index}] is not an object")
            continue
        try:
            relative_path = safe_manifest_path(entry.get("path"))
        except ValueError as error:
            failures.append(str(error))
            continue
        size_bytes = entry.get("size_bytes")
        digest = entry.get("sha256")
        if relative_path in recorded:
            failures.append(f"duplicate manifest path: {relative_path}")
            continue
        if not isinstance(size_bytes, int) or size_bytes < 0:
            failures.append(f"invalid size for {relative_path}")
            continue
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            failures.append(f"invalid sha256 for {relative_path}")
            continue
        recorded[relative_path] = (size_bytes, digest)

    if manifest.get("file_count") != len(recorded):
        failures.append(
            f"file_count {manifest.get('file_count')} does not match {len(recorded)} entries"
        )
    expected_total = sum(size for size, _ in recorded.values())
    if manifest.get("total_bytes") != expected_total:
        failures.append(
            f"total_bytes {manifest.get('total_bytes')} does not match {expected_total}"
        )

    actual_inventory = source_inventory(root)
    actual = {entry.path: (entry.size_bytes, entry.sha256) for entry in actual_inventory}
    for relative_path in sorted(set(recorded) - set(actual)):
        failures.append(f"manifest path missing from source payload: {relative_path}")
    for relative_path in sorted(set(actual) - set(recorded)):
        failures.append(f"source payload missing from manifest: {relative_path}")
    for relative_path in sorted(set(actual) & set(recorded)):
        if actual[relative_path] != recorded[relative_path]:
            failures.append(f"size or sha256 drift: {relative_path}")

    if checksum_path.is_file():
        try:
            checksums = parse_checksums(checksum_path)
        except ValueError as error:
            failures.append(str(error))
            checksums = {}
        expected_checksum_paths = set(actual) | {PACKAGE_MANIFEST_NAME}
        for relative_path in sorted(set(checksums) - expected_checksum_paths):
            failures.append(f"unexpected SHA256SUMS path: {relative_path}")
        for relative_path in sorted(expected_checksum_paths - set(checksums)):
            failures.append(f"missing SHA256SUMS path: {relative_path}")
        for relative_path in sorted(expected_checksum_paths & set(checksums)):
            target = root / relative_path
            if sha256_file(target) != checksums[relative_path]:
                failures.append(f"SHA256SUMS digest mismatch: {relative_path}")
    else:
        failures.append(f"missing {CHECKSUM_MANIFEST_NAME}")

    if arguments.strict_package:
        failures.extend(find_forbidden_packaged_paths(root))

    print(
        f"manifest entries {len(recorded)}, source payload files {len(actual)}, "
        f"payload bytes {sum(size for size, _ in actual.values())}"
    )
    if failures:
        for failure in failures:
            print(f"  FAIL {failure}")
        print(f"\nFAIL ({len(failures)})")
        return 1
    print("\npackage manifest, payload, and checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
