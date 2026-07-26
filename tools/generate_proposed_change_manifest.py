#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path


EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}
EXCLUDED_TOP_LEVEL_DIRECTORIES = {
    ".audit",
    "build",
}
EXCLUDED_PATHS = {
    "PACKAGE_MANIFEST.json",
    "SHA256SUMS",
    "docs/PROPOSED_CHANGE_MANIFEST.md",
}


@dataclass(frozen=True)
class FileIdentity:
    relative_path: str
    sha256: str
    size_bytes: int


def resolve_repository_root(candidate: Path) -> Path:
    candidate = candidate.resolve()
    if (candidate / "README.md").is_file():
        return candidate

    child_directories = [
        child
        for child in candidate.iterdir()
        if child.is_dir() and child.name not in EXCLUDED_DIRECTORY_NAMES
    ]
    repository_children = [
        child for child in child_directories if (child / "README.md").is_file()
    ]
    if len(repository_children) == 1:
        return repository_children[0]

    raise RuntimeError(f"cannot identify repository root below {candidate}")


def should_include(relative_path: Path) -> bool:
    if not relative_path.parts:
        return False
    excluded_directory_names = (
        EXCLUDED_DIRECTORY_NAMES | EXCLUDED_TOP_LEVEL_DIRECTORIES
    )
    if any(part in excluded_directory_names for part in relative_path.parts):
        return False
    if relative_path.as_posix() in EXCLUDED_PATHS:
        return False
    if relative_path.name == ".DS_Store" or relative_path.name.startswith("._"):
        return False
    if relative_path.name.endswith(
        (".pyc", ".o", ".a", ".so", ".dylib", ".dll", ".exe")
    ):
        return False
    return True


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def collect_file_identities(repository_root: Path) -> dict[str, FileIdentity]:
    identities: dict[str, FileIdentity] = {}
    for path in sorted(repository_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(repository_root)
        if not should_include(relative_path):
            continue
        relative_path_text = relative_path.as_posix()
        identities[relative_path_text] = FileIdentity(
            relative_path=relative_path_text,
            sha256=hash_file(path),
            size_bytes=path.stat().st_size,
        )
    return identities


def write_manifest(
    current_root: Path,
    baseline_root: Path,
    output_path: Path,
) -> None:
    current_files = collect_file_identities(current_root)
    baseline_files = collect_file_identities(baseline_root)

    current_paths = set(current_files)
    baseline_paths = set(baseline_files)
    added_paths = sorted(current_paths - baseline_paths)
    deleted_paths = sorted(baseline_paths - current_paths)
    modified_paths = sorted(
        path
        for path in current_paths & baseline_paths
        if current_files[path].sha256 != baseline_files[path].sha256
    )

    lines = [
        "# Proposed Change Manifest",
        "",
        "This is a hash-based inventory, not a textual diff. Generated build products,",
        "transient audit logs, object files, archives, Git metadata, and Python bytecode are excluded.",
        "Retained validation receipts under `docs/` are included.",
        "",
        "## Summary",
        "",
        f"- Added source/documentation files: **{len(added_paths)}**",
        f"- Modified source/documentation files: **{len(modified_paths)}**",
        f"- Deleted source/documentation files: **{len(deleted_paths)}**",
        "",
    ]

    def append_section(title: str, paths: list[str], source: dict[str, FileIdentity]) -> None:
        lines.extend((f"## {title}", ""))
        if not paths:
            lines.extend(("None.", ""))
            return
        lines.extend(("| Path | Bytes | SHA-256 |", "|---|---:|---|"))
        for relative_path in paths:
            identity = source[relative_path]
            lines.append(
                f"| `{relative_path}` | {identity.size_bytes} | `{identity.sha256}` |"
            )
        lines.append("")

    append_section("Added", added_paths, current_files)
    append_section("Modified", modified_paths, current_files)
    append_section("Deleted from Proposed Tree", deleted_paths, baseline_files)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a hash-only proposed-change inventory.")
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    current_root = resolve_repository_root(arguments.current)
    baseline_root = resolve_repository_root(arguments.baseline)
    write_manifest(current_root, baseline_root, arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
