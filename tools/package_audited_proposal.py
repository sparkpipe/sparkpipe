#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".audit",
    ".pytest_cache",
    "__MACOSX",
    "__pycache__",
    "build",
}
EXCLUDED_FILE_SUFFIXES = {
    ".a",
    ".dylib",
    ".dll",
    ".exe",
    ".o",
    ".obj",
    ".pyc",
    ".so",
    ".zip",
}
EXCLUDED_COMPOUND_SUFFIXES = {
    ".tar.gz",
    ".tar.bz2",
    ".tar.xz",
}
SENSITIVE_PATTERNS = (
    b"github" + b"_pat_",
    b"gh" + b"p_",
    b"gh" + b"o_",
    b"gh" + b"u_",
    b"gh" + b"s_",
    b"gh" + b"r_",
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN RSA " + b"PRIVATE KEY-----",
    b"-----BEGIN EC " + b"PRIVATE KEY-----",
    b"-----BEGIN OPENSSH " + b"PRIVATE KEY-----",
)


@dataclass(frozen=True)
class SourceSnapshotEntry:
    entry_type: str
    size_bytes: int
    sha256: str
    link_target: str


class ArchiveOutputLock:
    def __init__(self, output_path: Path) -> None:
        self.lock_path = output_path.parent / f".{output_path.name}.lock"
        self.lock_file: TextIO | None = None

    def __enter__(self) -> "ArchiveOutputLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self.lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self.lock_file.seek(0)
            owner = self.lock_file.read().strip()
            self.lock_file.close()
            self.lock_file = None
            owner_description = owner if owner else "unknown process"
            raise RuntimeError(
                f"another packager owns {self.lock_path}: "
                f"{owner_description}"
            ) from error

        self.lock_file.seek(0)
        self.lock_file.truncate()
        self.lock_file.write(f"pid={os.getpid()} operation=package-output\n")
        self.lock_file.flush()
        os.fsync(self.lock_file.fileno())
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self.lock_file is None:
            return
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        self.lock_file = None


class RepositorySnapshotLock:
    def __init__(self, repository_root: Path) -> None:
        self.lock_path = repository_root / ".audit" / "deep-validation.lock"
        self.lock_file: TextIO | None = None

    def __enter__(self) -> "RepositorySnapshotLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file = self.lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(
                self.lock_file.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError as error:
            self.lock_file.seek(0)
            owner = self.lock_file.read().strip()
            self.lock_file.close()
            self.lock_file = None
            owner_description = owner if owner else "unknown process"
            raise RuntimeError(
                f"repository is busy at {self.lock_path}: "
                f"{owner_description}"
            ) from error

        self.lock_file.seek(0)
        self.lock_file.truncate()
        self.lock_file.write(f"pid={os.getpid()} operation=package-source\n")
        self.lock_file.flush()
        os.fsync(self.lock_file.fileno())
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self.lock_file is None:
            return
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        self.lock_file = None


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def should_include(relative_path: Path) -> bool:
    if not relative_path.parts:
        return False
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in relative_path.parts):
        return False
    if relative_path.name == ".DS_Store" or relative_path.name.startswith("._"):
        return False
    lower_name = relative_path.name.lower()
    if any(lower_name.endswith(suffix) for suffix in EXCLUDED_COMPOUND_SUFFIXES):
        return False
    if relative_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return False
    return True


def ensure_internal_symlink(repository_root: Path, source_path: Path) -> None:
    resolved_target = source_path.resolve(strict=False)
    try:
        resolved_target.relative_to(repository_root)
    except ValueError as error:
        raise RuntimeError(
            f"refusing external symlink {source_path} -> "
            f"{os.readlink(source_path)}"
        ) from error


def collect_source_snapshot(
    repository_root: Path,
) -> dict[str, SourceSnapshotEntry]:
    snapshot: dict[str, SourceSnapshotEntry] = {}
    for source_path in sorted(repository_root.rglob("*")):
        relative_path = source_path.relative_to(repository_root)
        if not should_include(relative_path):
            continue
        relative_text = relative_path.as_posix()
        if source_path.is_symlink():
            ensure_internal_symlink(repository_root, source_path)
            snapshot[relative_text] = SourceSnapshotEntry(
                entry_type="symlink",
                size_bytes=0,
                sha256="",
                link_target=os.readlink(source_path),
            )
        elif source_path.is_dir():
            snapshot[relative_text] = SourceSnapshotEntry(
                entry_type="directory",
                size_bytes=0,
                sha256="",
                link_target="",
            )
        elif source_path.is_file():
            snapshot[relative_text] = SourceSnapshotEntry(
                entry_type="file",
                size_bytes=source_path.stat().st_size,
                sha256=hash_file(source_path),
                link_target="",
            )
        else:
            raise RuntimeError(f"unsupported source entry: {source_path}")
    return snapshot


def describe_snapshot_difference(
    before: dict[str, SourceSnapshotEntry],
    after: dict[str, SourceSnapshotEntry],
) -> str:
    before_paths = set(before)
    after_paths = set(after)
    added = sorted(after_paths - before_paths)
    removed = sorted(before_paths - after_paths)
    changed = sorted(
        path
        for path in before_paths & after_paths
        if before[path] != after[path]
    )
    parts: list[str] = []
    if added:
        parts.append("added=" + ",".join(added[:8]))
    if removed:
        parts.append("removed=" + ",".join(removed[:8]))
    if changed:
        parts.append("changed=" + ",".join(changed[:8]))
    return "; ".join(parts) if parts else "unknown difference"


def copy_source_tree(repository_root: Path, staging_root: Path) -> None:
    for source_path in sorted(repository_root.rglob("*")):
        relative_path = source_path.relative_to(repository_root)
        if not should_include(relative_path):
            continue
        destination_path = staging_root / relative_path
        if source_path.is_symlink():
            ensure_internal_symlink(repository_root, source_path)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            destination_path.symlink_to(os.readlink(source_path))
        elif source_path.is_dir():
            destination_path.mkdir(parents=True, exist_ok=True)
        elif source_path.is_file():
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        else:
            raise RuntimeError(f"unsupported source entry: {source_path}")


def verify_staged_source_snapshot(
    expected_snapshot: dict[str, SourceSnapshotEntry],
    staging_root: Path,
) -> None:
    staged_snapshot = collect_source_snapshot(staging_root)
    if staged_snapshot != expected_snapshot:
        difference = describe_snapshot_difference(
            expected_snapshot,
            staged_snapshot,
        )
        raise RuntimeError(
            "staged source does not match the locked source snapshot: "
            f"{difference}"
        )


def scan_for_sensitive_material(staging_root: Path) -> list[str]:
    findings: list[str] = []
    for path in sorted(staging_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        content = path.read_bytes()
        for pattern in SENSITIVE_PATTERNS:
            if pattern in content:
                findings.append(
                    f"{path.relative_to(staging_root).as_posix()}: "
                    "contains a sensitive credential pattern"
                )
    return findings


def normalize_permissions(staging_root: Path) -> None:
    for path in sorted(staging_root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
            continue
        current_mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(0o755 if current_mode & 0o111 else 0o644)


def write_internal_manifests(staging_root: Path) -> None:
    entries: list[dict[str, object]] = []
    checksum_lines: list[str] = []
    for path in sorted(staging_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(staging_root).as_posix()
        if relative_path in {"PACKAGE_MANIFEST.json", "SHA256SUMS"}:
            continue
        sha256 = hash_file(path)
        entries.append({
            "path": relative_path,
            "sha256": sha256,
            "size_bytes": path.stat().st_size,
        })
        checksum_lines.append(f"{sha256}  {relative_path}")
    manifest = {
        "schema_version": 1,
        "package_kind": "sparkpipe-audited-proposed-source-tree",
        "source_snapshot_has_git_history": False,
        "file_count": len(entries),
        "files": entries,
    }
    (staging_root / "PACKAGE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (staging_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n",
        encoding="utf-8",
    )


def normalized_tar_info(path: Path, arcname: str) -> tarfile.TarInfo:
    information = tarfile.TarInfo(arcname)
    information.uid = 0
    information.gid = 0
    information.uname = "root"
    information.gname = "root"
    information.mtime = 0
    if path.is_symlink():
        information.type = tarfile.SYMTYPE
        information.linkname = os.readlink(path)
        information.mode = 0o777
        information.size = 0
    elif path.is_dir():
        information.type = tarfile.DIRTYPE
        information.mode = 0o755
        information.size = 0
    else:
        information.type = tarfile.REGTYPE
        information.mode = stat.S_IMODE(path.stat().st_mode)
        information.size = path.stat().st_size
    return information


def create_deterministic_archive(
    staging_root: Path,
    output_path: Path,
    root_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                root_info = normalized_tar_info(staging_root, root_name)
                archive.addfile(root_info)
                for path in sorted(staging_root.rglob("*")):
                    relative_path = path.relative_to(staging_root).as_posix()
                    arcname = f"{root_name}/{relative_path}"
                    information = normalized_tar_info(path, arcname)
                    if path.is_file() and not path.is_symlink():
                        with path.open("rb") as input_file:
                            archive.addfile(information, input_file)
                    else:
                        archive.addfile(information)
        raw_output.flush()
        os.fsync(raw_output.fileno())


def verify_archive(output_path: Path, root_name: str) -> None:
    with tarfile.open(output_path, "r:gz") as archive:
        members = archive.getmembers()
        if not members or members[0].name != root_name:
            raise RuntimeError("archive root directory is missing")
        member_names: set[str] = set()
        for member in members:
            member_path = Path(member.name)
            if member.name in member_names:
                raise RuntimeError(f"duplicate archive member: {member.name}")
            member_names.add(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe archive member: {member.name}")
            if not (
                member.name == root_name
                or member.name.startswith(root_name + "/")
            ):
                raise RuntimeError(
                    f"member escapes archive root: {member.name}"
                )
            if not (
                member.isdir()
                or member.isfile()
                or member.issym()
            ):
                raise RuntimeError(
                    f"unsupported archive member type: {member.name}"
                )
            relative_parts = member_path.parts[1:]
            if any(
                part in EXCLUDED_DIRECTORY_NAMES for part in relative_parts
            ):
                raise RuntimeError(
                    f"excluded directory entered archive: {member.name}"
                )
            if relative_parts and not should_include(Path(*relative_parts)):
                raise RuntimeError(
                    f"excluded file entered archive: {member.name}"
                )
            if member.issym():
                link_path = Path(member.linkname)
                if link_path.is_absolute() or ".." in link_path.parts:
                    raise RuntimeError(
                        f"unsafe archive symlink: {member.name} -> "
                        f"{member.linkname}"
                    )


def create_temporary_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    return Path(temporary_name)


def fsync_directory(path: Path) -> None:
    directory_descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def atomic_write_text(path: Path, content: str) -> None:
    temporary_path = create_temporary_output_path(path)
    try:
        with temporary_path.open("w", encoding="utf-8") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def publish_archive_atomically(
    staging_root: Path,
    output_path: Path,
    root_name: str,
) -> str:
    temporary_output_path = create_temporary_output_path(output_path)
    try:
        create_deterministic_archive(
            staging_root,
            temporary_output_path,
            root_name,
        )
        verify_archive(temporary_output_path, root_name)
        archive_sha256 = hash_file(temporary_output_path)
        os.replace(temporary_output_path, output_path)
        fsync_directory(output_path.parent)
        verify_archive(output_path, root_name)
        if hash_file(output_path) != archive_sha256:
            raise RuntimeError("archive hash changed during atomic publication")
        return archive_sha256
    finally:
        if temporary_output_path.exists():
            temporary_output_path.unlink()


def path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic, source-clean SparkPipe proposal tarball."
        )
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--root-name",
        default="sparkpipe-audited-proposed",
    )
    arguments = parser.parse_args()

    repository_root = arguments.repository.resolve()
    output_path = arguments.output.resolve()
    if not (repository_root / "README.md").is_file():
        raise RuntimeError(f"not a SparkPipe repository: {repository_root}")
    if "/" in arguments.root_name or arguments.root_name in {"", ".", ".."}:
        raise RuntimeError("root name must be one safe path component")
    if path_is_within(output_path, repository_root):
        raise RuntimeError("archive output must be outside the source tree")

    with RepositorySnapshotLock(repository_root):
        with ArchiveOutputLock(output_path):
            source_snapshot_before = collect_source_snapshot(repository_root)
            with tempfile.TemporaryDirectory(
                prefix="sparkpipe-package-"
            ) as temporary_directory:
                staging_root = Path(temporary_directory) / arguments.root_name
                staging_root.mkdir(parents=True)
                copy_source_tree(repository_root, staging_root)
                source_snapshot_after = collect_source_snapshot(repository_root)
                if source_snapshot_after != source_snapshot_before:
                    difference = describe_snapshot_difference(
                        source_snapshot_before,
                        source_snapshot_after,
                    )
                    raise RuntimeError(
                        "source tree changed while packaging: " + difference
                    )
                verify_staged_source_snapshot(
                    source_snapshot_before,
                    staging_root,
                )
                findings = scan_for_sensitive_material(staging_root)
                if findings:
                    raise RuntimeError(
                        "sensitive material detected:\n" + "\n".join(findings)
                    )
                normalize_permissions(staging_root)
                write_internal_manifests(staging_root)
                normalize_permissions(staging_root)
                archive_sha256 = publish_archive_atomically(
                    staging_root,
                    output_path,
                    arguments.root_name,
                )

            sidecar_path = Path(str(output_path) + ".sha256")
            atomic_write_text(
                sidecar_path,
                f"{archive_sha256}  {output_path.name}\n",
            )

    print(json.dumps({
        "archive": str(output_path),
        "sha256": archive_sha256,
        "sha256_file": str(sidecar_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
