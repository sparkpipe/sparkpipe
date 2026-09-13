#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


FAMILY_MARKER = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:glm(?:52|5_2)?|qwen(?:36|3_6)?|dsv4|deepseek_v4|mimo(?:25|_2_5)?|k3|kimi_k3)(?:[^A-Za-z0-9]|$)",
    re.IGNORECASE,
)
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')
ASSIGNMENT_PATTERN = re.compile(
    r"^(SPARKPIPE_(?:CORE|COMPILER|RUNTIME)_SOURCES)\s*:=\s*(.*)$"
)
SOURCE_PATTERN = re.compile(r"([A-Za-z0-9_./-]+\.(?:c|cc|cpp|cxx))")


class AuditFailure(Exception):
    pass


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that SparkPipe core/compiler/runtime archives are model-neutral"
    )
    parser.add_argument("--repository", default=".")
    parser.add_argument("--core-archive", required=True)
    parser.add_argument("--compiler-archive", required=True)
    parser.add_argument("--runtime-archive", required=True)
    return parser.parse_args()


def parse_source_groups(manifest_path: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    current_name: str | None = None
    current_text: list[str] = []

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        match = ASSIGNMENT_PATTERN.match(line)
        if match is not None:
            if current_name is not None:
                raise AuditFailure(f"unterminated source assignment: {current_name}")
            current_name = match.group(1)
            current_text = [match.group(2)]
        elif current_name is not None:
            current_text.append(line)
        else:
            continue

        if current_name is not None and not line.endswith("\\"):
            joined = " ".join(part.rstrip("\\").strip() for part in current_text)
            groups[current_name] = SOURCE_PATTERN.findall(joined)
            current_name = None
            current_text = []

    if current_name is not None:
        raise AuditFailure(f"unterminated source assignment: {current_name}")

    required = {
        "SPARKPIPE_CORE_SOURCES",
        "SPARKPIPE_COMPILER_SOURCES",
        "SPARKPIPE_RUNTIME_SOURCES",
    }
    missing = required - groups.keys()
    if missing:
        raise AuditFailure(
            "sources.mk lacks neutral source groups: " + ", ".join(sorted(missing))
        )
    return groups


def resolve_quoted_include(
    repository: Path,
    source_path: Path,
    include_name: str,
) -> Path | None:
    candidates = (
        source_path.parent / include_name,
        repository / include_name,
        repository / "include" / include_name,
        repository / "src" / include_name,
        repository / "runtime" / include_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def collect_include_closure(
    repository: Path,
    source_paths: list[Path],
) -> tuple[set[Path], list[str]]:
    repository = repository.resolve()
    neutral_roots = tuple(
        path.resolve()
        for path in (
            repository / "include",
            repository / "src",
            repository / "runtime",
        )
    )
    pending = [path.resolve() for path in source_paths]
    visited: set[Path] = set()
    failures: list[str] = []

    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        relative = path.relative_to(repository)
        text = path.read_text(encoding="utf-8", errors="replace")

        if FAMILY_MARKER.search(relative.as_posix()):
            failures.append(f"model-family path in neutral closure: {relative}")
        if FAMILY_MARKER.search(text):
            failures.append(f"model-family marker in neutral closure: {relative}")

        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.lstrip().startswith('#include "'):
                continue
            match = INCLUDE_PATTERN.match(line)
            if match is None:
                continue
            include_name = match.group(1)
            resolved = resolve_quoted_include(repository, path, include_name)
            if resolved is None:
                failures.append(
                    f"unresolved quoted include {include_name!r} from "
                    f"{relative}:{line_number}"
                )
                continue
            if not any(resolved.is_relative_to(root) for root in neutral_roots):
                failures.append(
                    f"neutral include escapes neutral roots: {relative}:{line_number} "
                    f"-> {resolved.relative_to(repository)}"
                )
                continue
            pending.append(resolved)

    return visited, failures


def run_checked(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise AuditFailure(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def audit_archive(
    repository: Path,
    archive_path: str,
    expected_sources: list[str],
) -> tuple[list[str], int, int]:
    archive = (repository / archive_path).resolve()
    if not archive.is_file():
        raise AuditFailure(f"archive missing: {archive_path}")

    # macOS ar lists the ranlib table of contents as a member; GNU ar does not
    members = [
        line.strip()
        for line in run_checked(["ar", "t", str(archive)]).splitlines()
        if line.strip() and line.strip() != "__.SYMDEF SORTED"
    ]
    expected_members = {Path(source).with_suffix(".o").name for source in expected_sources}
    actual_members = set(members)
    failures: list[str] = []

    if len(members) != len(actual_members):
        failures.append(f"{archive_path}: duplicate archive member name")
    missing = sorted(expected_members - actual_members)
    unexpected = sorted(actual_members - expected_members)
    if missing:
        failures.append(f"{archive_path}: missing objects: {', '.join(missing)}")
    if unexpected:
        failures.append(f"{archive_path}: unexpected objects: {', '.join(unexpected)}")

    symbol_output = run_checked(["nm", "-g", "--defined-only", str(archive)])
    symbol_count = 0
    for line in symbol_output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        symbol_count += 1
        if FAMILY_MARKER.search(stripped):
            failures.append(
                f"{archive_path}: model-family symbol exported: {stripped}"
            )
    for member in members:
        if FAMILY_MARKER.search(member):
            failures.append(f"{archive_path}: model-family object present: {member}")

    return failures, len(members), symbol_count


def main() -> int:
    arguments = parse_arguments()
    repository = Path(arguments.repository).resolve()
    manifest_path = repository / "sources.mk"
    makefile_path = repository / "Makefile"
    failures: list[str] = []

    if not manifest_path.is_file() or not makefile_path.is_file():
        raise AuditFailure("repository does not contain sources.mk and Makefile")

    groups = parse_source_groups(manifest_path)
    archive_groups = {
        arguments.core_archive: groups["SPARKPIPE_CORE_SOURCES"],
        arguments.compiler_archive: groups["SPARKPIPE_COMPILER_SOURCES"],
        arguments.runtime_archive: groups["SPARKPIPE_RUNTIME_SOURCES"],
    }

    source_paths: list[Path] = []
    seen_sources: set[str] = set()
    for group_name in (
        "SPARKPIPE_CORE_SOURCES",
        "SPARKPIPE_COMPILER_SOURCES",
        "SPARKPIPE_RUNTIME_SOURCES",
    ):
        for source in groups[group_name]:
            if source in seen_sources:
                failures.append(f"neutral source appears in multiple groups: {source}")
            seen_sources.add(source)
            source_path = repository / source
            if not source_path.is_file():
                failures.append(f"neutral source missing: {source}")
                continue
            if FAMILY_MARKER.search(source):
                failures.append(f"model-family source listed as neutral: {source}")
            source_paths.append(source_path)

    closure, include_failures = collect_include_closure(repository, source_paths)
    failures.extend(include_failures)

    makefile_text = makefile_path.read_text(encoding="utf-8")
    required_makefile_fragments = (
        "CORE_INCLUDE_FLAGS := -I. -Iinclude -Isrc",
        "$(CORE_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)",
        "$(COMPILER_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)",
        "$(RUNTIME_OBJECTS): SP_INCLUDE_FLAGS = $(CORE_INCLUDE_FLAGS)",
        "sp_objects = $(patsubst %.c,build/obj/%.o,$(1))",
    )
    for fragment in required_makefile_fragments:
        if fragment not in makefile_text:
            failures.append(f"Makefile lacks neutral build boundary: {fragment}")

    archive_member_count = 0
    exported_symbol_count = 0
    for archive_path, expected_sources in archive_groups.items():
        archive_failures, member_count, symbol_count = audit_archive(
            repository,
            archive_path,
            expected_sources,
        )
        failures.extend(archive_failures)
        archive_member_count += member_count
        exported_symbol_count += symbol_count

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS core-boundary audit: "
        f"{len(source_paths)} neutral sources, "
        f"{len(closure)} source/header dependencies, "
        f"{archive_member_count} archive members, "
        f"{exported_symbol_count} exported symbols"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
