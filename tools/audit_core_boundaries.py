#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path

FAMILY_MARKER = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:glm52|qwen36|dsv4|mimo25|k3)(?:[^A-Za-z0-9]|$)",
    re.IGNORECASE)
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]')
SOURCE_PATTERN = re.compile(r"(?:^|\s)([A-Za-z0-9_./-]+\.(?:c|cc|cpp|cxx))(?=\s|\\|$)")


class AuditFailure(Exception):
    pass


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Verify that SparkPipe core archives and include closure are model-neutral")
    parser.add_argument("--repository", default=".")
    parser.add_argument("--core-archive", required=True)
    parser.add_argument("--compiler-archive", required=True)
    parser.add_argument("--runtime-archive", required=True)
    return parser.parse_args()


def read_manifest_sources(path):
    text = path.read_text()
    return [match.group(1) for match in SOURCE_PATTERN.finditer(text)]


def resolve_quoted_include(repository, source_path, include_name):
    candidates = [
        source_path.parent / include_name,
        repository / "include" / include_name,
        repository / "src" / include_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def collect_core_include_closure(repository, source_paths):
    allowed_roots = [
        (repository / "include").resolve(),
        (repository / "src").resolve(),
    ]
    pending = [path.resolve() for path in source_paths]
    visited = set()
    failures = []

    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        text = path.read_text(errors="replace")
        relative_path = path.relative_to(repository.resolve())
        if FAMILY_MARKER.search(str(relative_path)) or FAMILY_MARKER.search(text):
            failures.append(f"model-family marker in core closure: {relative_path}")

        for line_number, line in enumerate(text.splitlines(), 1):
            match = INCLUDE_PATTERN.match(line)
            if match is None:
                continue
            include_name = match.group(1)
            if not line.lstrip().startswith('#include "'):
                continue
            resolved = resolve_quoted_include(repository, path, include_name)
            if resolved is None:
                failures.append(
                    f"unresolved quoted include {include_name!r} from "
                    f"{relative_path}:{line_number}")
                continue
            if not any(resolved.is_relative_to(root) for root in allowed_roots):
                failures.append(
                    f"core include escapes neutral roots: {relative_path}:{line_number} "
                    f"-> {resolved.relative_to(repository.resolve())}")
                continue
            pending.append(resolved)

    return visited, failures


def run_checked(command):
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)
    if completed.returncode != 0:
        raise AuditFailure(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}")
    return completed.stdout


def audit_archive(repository, archive_path, expected_sources):
    archive = (repository / archive_path).resolve()
    if not archive.is_file():
        raise AuditFailure(f"archive missing: {archive_path}")

    archive_metadata_members = {"__.SYMDEF", "__.SYMDEF SORTED"}
    members = [
        line.strip()
        for line in run_checked(["ar", "t", str(archive)]).splitlines()
        if line.strip() and line.strip() not in archive_metadata_members
    ]
    expected_members = {Path(source).with_suffix(".o").name for source in expected_sources}
    actual_members = set(members)
    failures = []
    if actual_members != expected_members:
        missing = sorted(expected_members - actual_members)
        unexpected = sorted(actual_members - expected_members)
        if missing:
            failures.append(f"{archive_path}: missing objects: {', '.join(missing)}")
        if unexpected:
            failures.append(f"{archive_path}: unexpected objects: {', '.join(unexpected)}")

    symbol_output = run_checked(["nm", "-g", str(archive)])
    for line in symbol_output.splitlines():
        if FAMILY_MARKER.search(line):
            failures.append(f"{archive_path}: model-family symbol exported: {line.strip()}")
    for member in members:
        if FAMILY_MARKER.search(member):
            failures.append(f"{archive_path}: model-family object present: {member}")
    return failures


def main():
    arguments = parse_arguments()
    repository = Path(arguments.repository).resolve()
    manifest_path = repository / "core" / "sources.mk"
    makefile_path = repository / "Makefile"
    failures = []

    if not manifest_path.is_file() or not makefile_path.is_file():
        raise AuditFailure("repository does not contain core/sources.mk and Makefile")

    manifest_sources = read_manifest_sources(manifest_path)
    source_groups = {
        arguments.core_archive: [
            source for source in manifest_sources
            if Path(source).name in {
                "spark_status.c",
                "spark_filesystem.c",
                "spark_json.c",
                "spark_sha256.c",
            }
        ],
        arguments.compiler_archive: [
            source for source in manifest_sources
            if Path(source).name in {
                "spark_model_description.c",
                "spark_module_library.c",
                "spark_driver_compiler.c",
            }
        ],
        arguments.runtime_archive: [
            source for source in manifest_sources
            if Path(source).name in {
                "spark_driver_loader.c",
                "spark_orchestrator.c",
            }
        ],
    }

    if sum(len(group) for group in source_groups.values()) != len(manifest_sources):
        failures.append("core/sources.mk contains an unclassified source")

    source_paths = []
    for source in manifest_sources:
        source_path = repository / source
        if not source_path.is_file():
            failures.append(f"manifest source missing: {source}")
        else:
            source_paths.append(source_path)
        if not source.startswith("src/"):
            failures.append(f"core source is outside src/: {source}")
        if FAMILY_MARKER.search(source):
            failures.append(f"model-family source listed in core manifest: {source}")

    include_closure, include_failures = collect_core_include_closure(
        repository,
        source_paths)
    failures.extend(include_failures)

    makefile_text = makefile_path.read_text()
    required_makefile_fragments = [
        "CORE_INCLUDE_FLAGS := -Iinclude -Isrc",
        "build/core/%.o: src/%.c",
        "build/compiler/%.o: src/%.c",
        "build/runtime/%.o: src/%.c",
        "$(CC) $(CORE_INCLUDE_FLAGS) $(CFLAGS)",
    ]
    for fragment in required_makefile_fragments:
        if fragment not in makefile_text:
            failures.append(f"Makefile lacks isolated core build fragment: {fragment}")

    for relative_path in [repository / "include" / "sparkpipe", repository / "src"]:
        for path in relative_path.rglob("*"):
            if path.is_file() and FAMILY_MARKER.search(path.name):
                failures.append(
                    f"model-family file remains in neutral core tree: "
                    f"{path.relative_to(repository)}")

    for archive_path, expected_sources in source_groups.items():
        failures.extend(audit_archive(repository, archive_path, expected_sources))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS core-boundary audit: "
        f"{len(manifest_sources)} neutral sources, "
        f"{len(include_closure)} source/header files in include closure, "
        "no model-family archive members or exported symbols")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
