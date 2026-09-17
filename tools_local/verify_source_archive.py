#!/usr/bin/env python3
"""Safely extract and verify a deterministic SparkPipe source archive."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]


def validate_member(member: tarfile.TarInfo, expected_root: str | None) -> str:
    pure = PurePosixPath(member.name)
    if member.name.startswith("/") or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe archive path: {member.name}")
    if member.issym() or member.islnk() or member.isdev() or member.isfifo():
        raise ValueError(f"forbidden archive entry: {member.name}")
    if not member.isdir() and not member.isfile():
        raise ValueError(f"unsupported archive entry: {member.name}")
    root_name = pure.parts[0]
    if expected_root is not None and root_name != expected_root:
        raise ValueError(f"multiple archive roots: {expected_root}, {root_name}")
    return root_name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--extract-to", type=Path)
    arguments = parser.parse_args()

    archive_path = arguments.archive.resolve()
    temporary = None
    if arguments.extract_to is None:
        temporary = tempfile.TemporaryDirectory(prefix="sparkpipe-archive-verify-")
        extraction_root = Path(temporary.name)
    else:
        extraction_root = arguments.extract_to.resolve()
        if extraction_root.exists():
            shutil.rmtree(extraction_root)
        extraction_root.mkdir(parents=True)

    expected_root = None
    member_names: set[str] = set()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            if member.name in member_names:
                raise SystemExit(f"duplicate archive member: {member.name}")
            member_names.add(member.name)
            expected_root = validate_member(member, expected_root)
        if expected_root is None:
            raise SystemExit("archive is empty")
        archive.extractall(extraction_root, members=members, filter="data")

    package_root = extraction_root / expected_root
    verifier = package_root / "tools" / "verify_package_manifest.py"
    verifier_environment = os.environ.copy()
    verifier_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            "python3",
            "-B",
            str(verifier),
            "--root",
            str(package_root),
            "--strict-package",
        ],
        check=False,
        text=True,
        env=verifier_environment,
    )
    if temporary is not None:
        temporary.cleanup()
    if completed.returncode != 0:
        return completed.returncode
    print(f"archive verified with root {expected_root} and {len(members)} members")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
