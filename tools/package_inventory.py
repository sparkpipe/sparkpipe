#!/usr/bin/env python3
"""Deterministic source-package inventory shared by generation and verification."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator


PACKAGE_MANIFEST_NAME = "PACKAGE_MANIFEST.json"
CHECKSUM_MANIFEST_NAME = "SHA256SUMS"
EVIDENCE_MANIFEST_NAME = "EVIDENCE_MANIFEST.json"
METADATA_NAMES = {
    PACKAGE_MANIFEST_NAME,
    CHECKSUM_MANIFEST_NAME,
    EVIDENCE_MANIFEST_NAME,
}
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "build",
}
EXCLUDED_DIRECTORY_PREFIXES = (
    "docs/validation-logs/",
    "qualification/raw/",
    "qualification/receipts/",
    "qualification/runs/",
    "qualification/evidence/",
    "qualification/ds4_eval/",
)
EXCLUDED_FILE_SUFFIXES = {
    ".a",
    ".dylib",
    ".dll",
    ".o",
    ".obj",
    ".pyc",
    ".pyo",
    ".so",
}
# Local secrets never ship, tracked or not: a source package that walks the
# filesystem (Git-independent by design) will otherwise pick up whatever
# credentials a developer's checkout happens to hold. This exact case was
# caught by the verify gate after a local .env landed in SHA256SUMS.
EXCLUDED_FILE_NAMES = {
    ".env",
}
QUALIFICATION_EVIDENCE_SUFFIXES = {
    ".log",
    ".receipt",
    ".latest.txt",
}
FORBIDDEN_ARCHIVE_SUFFIXES = (
    ".7z",
    ".deb",
    ".rpm",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".whl",
    ".zip",
)


@dataclass(frozen=True)
class PackageFile:
    path: str
    size_bytes: int
    sha256: str


def normalize_relative_path(path: Path) -> str:
    value = path.as_posix()
    pure = PurePosixPath(value)
    if value == "" or value.startswith("/"):
        raise ValueError(f"invalid package path: {value!r}")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe package path: {value!r}")
    return value


def is_forbidden_archive_path(relative_path: str) -> bool:
    lowered = relative_path.lower()
    return any(lowered.endswith(suffix) for suffix in FORBIDDEN_ARCHIVE_SUFFIXES)


def has_excluded_directory_prefix(relative_path: str) -> bool:
    return any(
        relative_path == prefix.rstrip("/") or relative_path.startswith(prefix)
        for prefix in EXCLUDED_DIRECTORY_PREFIXES
    )


def is_qualification_evidence_file(relative_path: str) -> bool:
    pure = PurePosixPath(relative_path)
    if not relative_path.startswith("qualification/"):
        return False
    return (
        pure.suffix.lower() in QUALIFICATION_EVIDENCE_SUFFIXES
        or pure.name.lower().endswith(".latest.txt")
        or pure.name.lower().startswith("evidence.")
    )


def is_excluded_relative_path(relative_path: str) -> bool:
    pure = PurePosixPath(relative_path)
    if relative_path in METADATA_NAMES:
        return True
    if pure.name in EXCLUDED_FILE_NAMES:
        return True
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in pure.parts):
        return True
    if has_excluded_directory_prefix(relative_path):
        return True
    if is_qualification_evidence_file(relative_path):
        return True
    if pure.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    if is_forbidden_archive_path(relative_path):
        return True
    return False


def iter_source_paths(root: Path) -> Iterator[Path]:
    root = root.resolve()
    for directory_path, directory_names, file_names in os.walk(root):
        directory = Path(directory_path)
        retained_directories = []
        for name in sorted(directory_names):
            child = directory / name
            relative_path = normalize_relative_path(child.relative_to(root))
            if child.is_symlink():
                raise ValueError(f"symbolic links are forbidden: {relative_path}")
            if name in EXCLUDED_DIRECTORY_NAMES:
                continue
            if has_excluded_directory_prefix(relative_path):
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for file_name in sorted(file_names):
            absolute_path = directory / file_name
            relative_path = normalize_relative_path(absolute_path.relative_to(root))
            if is_excluded_relative_path(relative_path):
                continue
            if absolute_path.is_symlink():
                raise ValueError(f"symbolic links are forbidden: {relative_path}")
            if not absolute_path.is_file():
                raise ValueError(f"non-regular package entry: {relative_path}")
            yield absolute_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory(root: Path) -> list[PackageFile]:
    resolved_root = root.resolve()
    inventory = []
    for absolute_path in iter_source_paths(resolved_root):
        relative_path = normalize_relative_path(absolute_path.relative_to(resolved_root))
        inventory.append(
            PackageFile(
                path=relative_path,
                size_bytes=absolute_path.stat().st_size,
                sha256=sha256_file(absolute_path),
            )
        )
    inventory.sort(key=lambda entry: entry.path)
    return inventory


def find_forbidden_packaged_paths(root: Path) -> list[str]:
    root = root.resolve()
    failures: list[str] = []
    for absolute_path in sorted(root.rglob("*")):
        relative_path = normalize_relative_path(absolute_path.relative_to(root))
        pure = PurePosixPath(relative_path)
        if absolute_path.is_symlink():
            failures.append(f"symbolic link: {relative_path}")
            continue
        if any(part in EXCLUDED_DIRECTORY_NAMES for part in pure.parts):
            failures.append(f"excluded build/cache path: {relative_path}")
            continue
        if has_excluded_directory_prefix(relative_path) or is_qualification_evidence_file(relative_path):
            failures.append(f"qualification evidence in source package: {relative_path}")
            continue
        if not absolute_path.is_file():
            continue
        if pure.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
            failures.append(f"compiled artifact in source package: {relative_path}")
            continue
        if is_forbidden_archive_path(relative_path):
            failures.append(f"nested archive in source package: {relative_path}")
    return failures


def inventory_paths(entries: Iterable[PackageFile]) -> set[str]:
    return {entry.path for entry in entries}
