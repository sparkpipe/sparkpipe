#!/usr/bin/env python3
"""Provider-neutral local coding harness for raced Ox Alpha inference.

Codex owns scheduling and integration.  This module owns one durable logical
agent session: it sends each model turn through a provider race, executes a
small set of workspace-confined tools exactly once, and journals raw responses
and tool results locally.  No upstream provider owns the conversation.
"""

from __future__ import annotations

import dataclasses
import ctypes
import errno
import fnmatch
import hashlib
import json
import os
import platform
import re
import secrets
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence


SESSION_ID_RE = re.compile(r"^codex-oxalpha-[0-9a-f]{32}$")
SECRET_ENV_RE = re.compile(
    r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|ACCESS_KEY|PRIVATE_KEY|CREDENTIAL)",
    re.IGNORECASE,
)
PROTECTED_FILE_RE = re.compile(r"^(?:\.env(?:\..*)?|.*\.(?:pem|key))$", re.IGNORECASE)
MAX_PATCH_BYTES = 4 << 20
MAX_COMMAND_BYTES = 4 << 20
MAX_READ_BYTES = 1 << 20
MAX_SEARCH_BYTES = 64 << 20
MAX_CONTEXT_TOOL_BYTES = 32 << 10
MAX_SESSION_CONTEXT_BYTES = 16 << 20
MAX_SESSION_ARTIFACT_BYTES = 1 << 30
MAX_EVENT_OUTPUT_BYTES = 4 << 20
MAX_STORAGE_ENTRIES = 262144
MAX_PROCESS_DESCENDANTS = 64
MAX_TOOL_CALLS_PER_TURN = 128
TOOL_COMPLETION_SCHEMA_VERSION = 2
EventCallback = Callable[[str, dict[str, Any] | None, int | None], None]
TRUSTED_EXECUTABLE_PATHS = {
    "bash": (Path("/bin/bash"),),
    "bwrap": (Path("/usr/bin/bwrap"), Path("/bin/bwrap")),
    "git": (Path("/usr/bin/git"),),
    "python3": (Path("/usr/bin/python3"),),
    "pwd": (Path("/bin/pwd"), Path("/usr/bin/pwd")),
    "sandbox-exec": (Path("/usr/bin/sandbox-exec"),),
}

PROCESS_GATE_CODE = """\
import os
import resource
import sys

gate = int(sys.argv[1])
ready_fd = int(sys.argv[2])
maximum = int(sys.argv[3])
process_limit = int(sys.argv[4])
try:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    current, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    if hard == resource.RLIM_INFINITY:
        hard = maximum
    hard = min(maximum, hard)
    resource.setrlimit(resource.RLIMIT_FSIZE, (hard, hard))
    current, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    if hard == resource.RLIM_INFINITY:
        hard = process_limit
    resource.setrlimit(resource.RLIMIT_NPROC, (min(process_limit, hard), min(process_limit, hard)))
    os.write(ready_fd, b"1")
finally:
    os.close(ready_fd)
try:
    ready = os.read(gate, 1)
finally:
    os.close(gate)
if ready != b"1":
    raise SystemExit(125)
os.execve(sys.argv[5], sys.argv[5:], os.environ)
"""

SEARCH_HELPER_CODE = r"""
import fnmatch
import json
import os
import re
import stat
import sys

query, target, pattern, fixed, limit, maximum_file, maximum_scan = json.loads(sys.argv[1])
expression = None if fixed else re.compile(query)
workspace = os.path.realpath(os.getcwd())
root = os.path.realpath(target)
if os.path.commonpath((workspace, root)) != workspace:
    raise SystemExit("search target escapes workspace")
protected = re.compile(r"^(?:\.env(?:\..*)?|.*\.(?:pem|key))$", re.IGNORECASE)
pending = [root]
matches = []
scanned = 0
truncated = False
while pending and not truncated:
    current = pending.pop()
    metadata = os.lstat(current)
    if stat.S_ISLNK(metadata.st_mode):
        continue
    if stat.S_ISDIR(metadata.st_mode):
        with os.scandir(current) as stream:
            entries = sorted(stream, key=lambda item: os.fsencode(item.name), reverse=True)
        for entry in entries:
            relative = os.path.relpath(entry.path, workspace)
            parts = relative.split(os.sep)
            if ".git" in parts or any(protected.fullmatch(part) for part in parts):
                continue
            pending.append(entry.path)
        continue
    relative = os.path.relpath(current, workspace).replace(os.sep, "/")
    if pattern and not fnmatch.fnmatchcase(relative, pattern):
        continue
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum_file:
        continue
    scanned += metadata.st_size
    if scanned > maximum_scan:
        truncated = True
        break
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(current, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
        ):
            raise SystemExit("search file changed while opening")
        data = os.read(descriptor, maximum_file + 1)
    finally:
        os.close(descriptor)
    if len(data) > maximum_file or b"\0" in data:
        continue
    for line_number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        found = query in line if fixed else expression.search(line) is not None
        if not found:
            continue
        matches.append(f"{relative}:{line_number}:{line[:4096]}")
        if len(matches) > limit:
            truncated = True
            break
print(json.dumps({"ok": True, "matches": matches[:limit], "truncated": truncated}, separators=(",", ":")))
"""


class HarnessError(RuntimeError):
    """A local harness invariant or policy was violated."""


@dataclasses.dataclass
class HarnessRunResult:
    exit_code: int
    text: str
    output: str
    session_id: str | None
    tokens: int
    malformed_json_lines: int = 0
    timed_out: bool = False
    test_receipts: tuple[dict[str, Any], ...] = ()


@dataclasses.dataclass
class ProcessResult:
    args: list[str]
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool = False
    stderr_truncated: bool = False


class _DarwinProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32),
        ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32),
        ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32),
        ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32),
        ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32),
        ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32),
        ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16),
        ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32),
        ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32),
        ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32),
        ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64),
        ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class _DarwinProcFdInfo(ctypes.Structure):
    _fields_ = [("proc_fd", ctypes.c_int32), ("proc_fdtype", ctypes.c_uint32)]


class _DarwinProcFileInfo(ctypes.Structure):
    _fields_ = [
        ("fi_openflags", ctypes.c_uint32),
        ("fi_status", ctypes.c_uint32),
        ("fi_offset", ctypes.c_int64),
        ("fi_type", ctypes.c_int32),
        ("fi_guardflags", ctypes.c_uint32),
    ]


class _DarwinVinfoStat(ctypes.Structure):
    _fields_ = [
        ("vst_dev", ctypes.c_uint32),
        ("vst_mode", ctypes.c_uint16),
        ("vst_nlink", ctypes.c_uint16),
        ("vst_ino", ctypes.c_uint64),
        ("vst_uid", ctypes.c_uint32),
        ("vst_gid", ctypes.c_uint32),
        ("vst_atime", ctypes.c_int64),
        ("vst_atimensec", ctypes.c_int64),
        ("vst_mtime", ctypes.c_int64),
        ("vst_mtimensec", ctypes.c_int64),
        ("vst_ctime", ctypes.c_int64),
        ("vst_ctimensec", ctypes.c_int64),
        ("vst_birthtime", ctypes.c_int64),
        ("vst_birthtimensec", ctypes.c_int64),
        ("vst_size", ctypes.c_int64),
        ("vst_blocks", ctypes.c_int64),
        ("vst_blksize", ctypes.c_int32),
        ("vst_flags", ctypes.c_uint32),
        ("vst_gen", ctypes.c_uint32),
        ("vst_rdev", ctypes.c_uint32),
        ("vst_qspare", ctypes.c_int64 * 2),
    ]


class _DarwinVnodeInfo(ctypes.Structure):
    _fields_ = [
        ("vi_stat", _DarwinVinfoStat),
        ("vi_type", ctypes.c_int32),
        ("vi_pad", ctypes.c_int32),
        ("vi_fsid", ctypes.c_int32 * 2),
    ]


class _DarwinVnodeInfoPath(ctypes.Structure):
    _fields_ = [("vip_vi", _DarwinVnodeInfo), ("vip_path", ctypes.c_char * 1024)]


class _DarwinProcVnodePathInfo(ctypes.Structure):
    _fields_ = [("pvi_cdir", _DarwinVnodeInfoPath), ("pvi_rdir", _DarwinVnodeInfoPath)]


class _DarwinVnodeFdInfo(ctypes.Structure):
    _fields_ = [("pfi", _DarwinProcFileInfo), ("vnode", _DarwinVnodeInfo)]


def trusted_executable(name: str) -> str:
    for path in TRUSTED_EXECUTABLE_PATHS.get(name, ()):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if (
            stat.S_ISREG(metadata.st_mode)
            and not path.is_symlink()
            and metadata.st_uid == 0
            and metadata.st_mode & 0o022 == 0
        ):
            return str(path)
    raise HarnessError(f"trusted executable is unavailable: {name}")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def path_allowed(path: str, patterns: Sequence[str]) -> bool:
    candidate = PurePosixPath(path)
    if candidate.is_absolute() or ".." in candidate.parts or ".git" in candidate.parts:
        return False
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        return False
    for raw_pattern in patterns:
        pattern = PurePosixPath(raw_pattern)
        if pattern.is_absolute() or ".." in pattern.parts:
            continue
        rendered = pattern.as_posix()
        if fnmatch.fnmatchcase(normalized, rendered):
            return True
        if rendered.endswith("/**") and normalized == rendered[:-3].rstrip("/"):
            return True
    return False


def _bounded_text(value: bytes, limit: int) -> tuple[str, bool]:
    truncated = len(value) > limit
    chunk = value[:limit]
    return chunk.decode("utf-8", errors="replace"), truncated


def _bounded_event_output(lines: Sequence[str], suffix: str = "") -> str:
    value, _truncated = _bounded_text(
        ("".join(lines) + suffix).encode("utf-8"),
        MAX_EVENT_OUTPUT_BYTES,
    )
    return value


def _session_parent_fd(root: Path, relative: Path) -> tuple[int, str]:
    candidate = PurePosixPath(relative.as_posix())
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise HarnessError(f"unsafe session archive path: {relative}")
    _validate_session_tree(root)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise HarnessError(f"session root is missing, symlinked, or unsafe: {root}") from error
    try:
        opened = os.fstat(descriptor)
        current = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise HarnessError(f"session root changed while opening: {root}")
        for part in candidate.parts[:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            current = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                os.close(child)
                raise HarnessError(f"session archive parent changed while opening: {relative}")
            os.close(descriptor)
            descriptor = child
        return descriptor, candidate.parts[-1]
    except (OSError, HarnessError) as error:
        os.close(descriptor)
        if isinstance(error, HarnessError):
            raise
        raise HarnessError(f"session archive parent is missing, symlinked, or unsafe: {relative}") from error


def _reject_symlink_components(path: Path) -> None:
    lexical = path if path.is_absolute() else Path.cwd() / path
    if ".." in lexical.parts:
        raise HarnessError(f"session path has an unsafe parent component: {path}")
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        if part in ("", "."):
            continue
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            break
        except OSError as error:
            raise HarnessError(f"cannot inspect session path component: {current}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise HarnessError(f"session path contains a symlink: {current}")


def _validate_session_tree(root: Path) -> None:
    _reject_symlink_components(root)
    try:
        metadata = root.lstat()
    except FileNotFoundError as error:
        raise HarnessError(f"session root does not exist: {root}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise HarnessError(f"session root is symlinked or not a directory: {root}")
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as stream:
                entries = list(stream)
        except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
            raise HarnessError(f"session storage changed or is unreadable: {current}") from error
        for entry in entries:
            try:
                item = entry.stat(follow_symlinks=False)
            except (FileNotFoundError, PermissionError) as error:
                raise HarnessError(f"session storage changed or is unreadable: {entry.path}") from error
            if stat.S_ISLNK(item.st_mode):
                raise HarnessError(f"session storage contains a symlink: {entry.path}")
            if stat.S_ISDIR(item.st_mode):
                pending.append(Path(entry.path))
            elif not stat.S_ISREG(item.st_mode):
                raise HarnessError(f"session storage contains a special file: {entry.path}")


def _session_mkdir(root: Path, relative: Path) -> None:
    descriptor, name = _session_parent_fd(root, relative)
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=descriptor)
            os.fsync(descriptor)
        except FileExistsError:
            pass
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise HarnessError(f"session directory is symlinked or unsafe: {relative}")
    except OSError as error:
        raise HarnessError(f"cannot create safe session directory: {relative}") from error
    finally:
        os.close(descriptor)


def _reject_existing_symlink(descriptor: int, name: str) -> None:
    try:
        metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise HarnessError(f"session archive target is symlinked or not regular: {name}")


def _session_atomic_bytes(root: Path, relative: Path, encoded: bytes) -> None:
    descriptor, name = _session_parent_fd(root, relative)
    temporary = f".{name}.{secrets.token_hex(8)}.tmp"
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(child, "wb") as stream:
            child = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, name, src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.fsync(descriptor)
    finally:
        if child >= 0:
            os.close(child)
        try:
            os.unlink(temporary, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        os.close(descriptor)


def _session_exclusive_bytes(root: Path, relative: Path, encoded: bytes) -> None:
    descriptor, name = _session_parent_fd(root, relative)
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=descriptor,
        )
        with os.fdopen(child, "wb") as stream:
            child = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(descriptor)
    finally:
        if child >= 0:
            os.close(child)
        os.close(descriptor)


def _session_read_bytes(root: Path, relative: Path) -> bytes:
    descriptor, name = _session_parent_fd(root, relative)
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        with os.fdopen(child, "rb") as stream:
            child = -1
            return stream.read()
    finally:
        if child >= 0:
            os.close(child)
        os.close(descriptor)


def _session_open_read_fd(root: Path, relative: Path) -> int:
    descriptor, name = _session_parent_fd(root, relative)
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        metadata = os.fstat(child)
        if not stat.S_ISREG(metadata.st_mode):
            raise HarnessError(f"session marker is not a regular file: {relative}")
        answer = child
        child = -1
        return answer
    finally:
        if child >= 0:
            os.close(child)
        os.close(descriptor)


def _session_open_regular_nofollow(root: Path, relative: Path) -> int:
    candidate = PurePosixPath(relative.as_posix())
    if candidate.is_absolute() or not candidate.parts or ".." in candidate.parts:
        raise HarnessError(f"unsafe session ownership path: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    child = -1
    try:
        descriptor = os.open(root, flags)
        opened = os.fstat(descriptor)
        current = os.stat(root, follow_symlinks=False)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise HarnessError("session ownership root changed while opening")
        for part in candidate.parts[:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            opened = os.fstat(child)
            current = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
            ):
                os.close(child)
                child = -1
                raise HarnessError(f"session ownership parent changed: {relative}")
            os.close(descriptor)
            descriptor = child
            child = -1
        child = os.open(
            candidate.parts[-1],
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=descriptor,
        )
        opened = os.fstat(child)
        current = os.stat(candidate.parts[-1], dir_fd=descriptor, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            os.close(child)
            child = -1
            raise HarnessError(f"session ownership file changed: {relative}")
        answer = child
        child = -1
        return answer
    except OSError as error:
        raise HarnessError(f"session ownership path is missing, symlinked, or unsafe: {relative}") from error
    finally:
        if child >= 0:
            os.close(child)
        if descriptor >= 0:
            os.close(descriptor)


def _session_append_record(root: Path, relative: Path, encoded: bytes) -> None:
    descriptor, name = _session_parent_fd(root, relative)
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(
            name,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=descriptor,
        )
        written = os.write(child, encoded)
        if written != len(encoded):
            raise HarnessError("journal record append was partial")
        os.fsync(child)
        os.fsync(descriptor)
    finally:
        if child >= 0:
            os.close(child)
        os.close(descriptor)


def _session_truncate(root: Path, relative: Path, length: int) -> None:
    descriptor, name = _session_parent_fd(root, relative)
    child = -1
    try:
        _reject_existing_symlink(descriptor, name)
        child = os.open(name, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=descriptor)
        os.ftruncate(child, length)
        os.fsync(child)
        os.fsync(descriptor)
    finally:
        if child >= 0:
            os.close(child)
        os.close(descriptor)


class WorkspaceTools:
    """Single-threaded tools scoped to one isolated agent workspace."""

    def __init__(
        self,
        workspace: Path,
        role: str,
        task: dict[str, Any],
        session_root: Path,
        *,
        command_timeout_seconds: int = 900,
        max_session_bytes: int = MAX_SESSION_ARTIFACT_BYTES,
        max_storage_entries: int = MAX_STORAGE_ENTRIES,
        max_process_descendants: int = MAX_PROCESS_DESCENDANTS,
        overall_deadline: float | None = None,
        process_event_callback: Callable[[dict[str, Any], int | None], None] | None = None,
    ):
        self.workspace = workspace.resolve()
        if role not in {"implementer", "auditor"}:
            raise HarnessError(f"invalid harness role: {role}")
        self.role = role
        self.task = task
        self.git_executable = trusted_executable("git")
        self.pwd_executable = trusted_executable("pwd")
        self.python_executable = trusted_executable("python3")
        _reject_symlink_components(session_root)
        absolute_session_root = session_root.absolute()
        raw_session_root = absolute_session_root.parent.resolve() / absolute_session_root.name
        if raw_session_root.is_symlink() or not raw_session_root.is_dir():
            raise HarnessError("session root must be an existing non-symlink directory")
        self.session_root = raw_session_root
        for parent, child in (
            (self.workspace, self.session_root),
            (self.session_root, self.workspace),
        ):
            try:
                child.relative_to(parent)
            except ValueError:
                continue
            raise HarnessError("workspace and session storage must be disjoint")
        self.command_timeout_seconds = command_timeout_seconds
        self.max_session_bytes = max_session_bytes
        self.max_storage_entries = max_storage_entries
        self.max_process_descendants = max_process_descendants
        self.overall_deadline = overall_deadline
        self.process_event_callback = process_event_callback
        self.tool_context: dict[str, Any] | None = None
        if (
            self.max_session_bytes <= 0
            or self.max_storage_entries <= 0
            or self.max_process_descendants <= 0
        ):
            raise HarnessError("storage budgets must be positive")
        self.sandbox_home = self.session_root / "sandbox-home"
        self.sandbox_tmp = self.session_root / "sandbox-tmp"
        self.process_root = self.session_root / "processes"
        for relative in (
            Path("processes"),
            Path("sandbox-home"),
            Path("sandbox-home/.config"),
            Path("sandbox-home/.cache"),
            Path("sandbox-home/.local"),
            Path("sandbox-home/.local/share"),
            Path("sandbox-tmp"),
        ):
            _session_mkdir(self.session_root, relative)
        _validate_session_tree(self.session_root)
        self._recover_owned_processes()
        baseline_path = self.session_root / "workspace-baseline.json"
        if baseline_path.exists():
            baseline = json.loads(
                _session_read_bytes(self.session_root, Path("workspace-baseline.json"))
            )
            if (
                baseline.get("workspace") != str(self.workspace)
                or baseline.get("role") != self.role
                or not isinstance(baseline.get("bytes"), int)
                or not isinstance(baseline.get("entries"), int)
            ):
                raise HarnessError("invalid persisted workspace storage baseline")
            self.workspace_baseline_bytes = baseline["bytes"]
            self.workspace_baseline_entries = baseline["entries"]
            self.auditor_baseline_fingerprint = baseline.get("fingerprint")
            if self.role == "auditor" and not isinstance(
                self.auditor_baseline_fingerprint, str
            ):
                raise HarnessError("auditor workspace baseline has no durable fingerprint")
        else:
            self.workspace_baseline_bytes, self.workspace_baseline_entries = self._tree_usage(
                self.workspace,
                (1 << 63) - 1,
                (1 << 63) - 1,
                excluded=(self.workspace / ".git",),
            )
            self.auditor_baseline_fingerprint = None
            if self.role == "auditor":
                self.auditor_baseline_fingerprint = self.workspace_fingerprint()
            _session_atomic_bytes(
                self.session_root,
                Path("workspace-baseline.json"),
                (canonical_json({
                    "workspace": str(self.workspace),
                    "role": self.role,
                    "bytes": self.workspace_baseline_bytes,
                    "entries": self.workspace_baseline_entries,
                    "fingerprint": self.auditor_baseline_fingerprint,
                }) + "\n").encode(),
            )
        if self.role == "auditor":
            self._assert_auditor_unchanged()

    def schemas(self) -> list[dict[str, Any]]:
        tools = [
            self._schema(
                "read_file",
                "Read a UTF-8 repository file by line range. Use repeated ranges for large files.",
                {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "line_count": {"type": "integer", "minimum": 1, "maximum": 2000},
                },
                ["path"],
            ),
            self._schema(
                "list_files",
                "List tracked and untracked repository files matching an optional glob.",
                {
                    "glob": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                },
                [],
            ),
            self._schema(
                "search_files",
                "Search repository text in-process; results are bounded and line-numbered.",
                {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "fixed_strings": {"type": "boolean"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 500},
                },
                ["query"],
            ),
            self._schema(
                "run_command",
                "Run a declared task test command or a read-only inspection command.",
                {"command": {"type": "string"}},
                ["command"],
            ),
            self._schema(
                "read_artifact",
                "Read another byte range from a raw tool-result artifact in this session.",
                {
                    "artifact": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "max_bytes": {"type": "integer", "minimum": 1, "maximum": 65536},
                },
                ["artifact"],
            ),
        ]
        if self.role == "implementer":
            tools.append(
                self._schema(
                    "apply_patch",
                    "Apply one git-apply-compatible unified patch inside the task's declared "
                    "write_set. Start with 'diff --git'; do not use '*** Begin Patch' syntax.",
                    {"patch": {"type": "string"}},
                    ["patch"],
                )
            )
            tools.append(
                self._schema(
                    "finish_task",
                    "Submit the final implementer contract after all edits and declared tests. "
                    "Call this as the only tool in the final turn.",
                    {
                        "status": {"type": "string", "enum": ["READY_FOR_AUDIT"]},
                        "summary": {"type": "string"},
                        "changed_paths": {"type": "array", "items": {"type": "string"}},
                        "tests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "exit_code": {"type": "integer"},
                                    "evidence": {"type": "string"},
                                },
                                "required": ["command", "exit_code", "evidence"],
                                "additionalProperties": False,
                            },
                        },
                        "known_limits": {"type": "array", "items": {"type": "string"}},
                        "hardware_claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "claim": {"type": "string"},
                                    "class": {
                                        "type": "string",
                                        "enum": ["MEASURED", "SIMULATED", "ANALYTICAL", "UNVERIFIED"],
                                    },
                                },
                                "required": ["claim", "class"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    ["status", "summary", "changed_paths", "tests", "known_limits", "hardware_claims"],
                )
            )
        else:
            tools.append(
                self._schema(
                    "finish_task",
                    "Submit the final auditor contract after all checks and declared tests. "
                    "Call this as the only tool in the final turn.",
                    {
                        "verdict": {
                            "type": "string",
                            "enum": ["APPROVE", "REJECT", "BLOCKED"],
                        },
                        "patch_sha256": {"type": "string"},
                        "findings": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "severity": {
                                        "type": "string",
                                        "enum": ["P0", "P1", "P2", "P3"],
                                    },
                                    "path": {"type": "string"},
                                    "line": {"type": "integer"},
                                    "title": {"type": "string"},
                                    "evidence": {"type": "string"},
                                },
                                "required": ["severity", "path", "line", "title", "evidence"],
                                "additionalProperties": False,
                            },
                        },
                        "tests": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "command": {"type": "string"},
                                    "exit_code": {"type": "integer"},
                                    "evidence": {"type": "string"},
                                },
                                "required": ["command", "exit_code", "evidence"],
                                "additionalProperties": False,
                            },
                        },
                        "scope_verified": {"type": "boolean"},
                        "tracked_source_unchanged_by_auditor": {"type": "boolean"},
                    },
                    [
                        "verdict",
                        "patch_sha256",
                        "findings",
                        "tests",
                        "scope_verified",
                        "tracked_source_unchanged_by_auditor",
                    ],
                )
            )
        return tools

    @staticmethod
    def _schema(
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str],
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        methods = {
            "read_file": self._read_file,
            "list_files": self._list_files,
            "search_files": self._search_files,
            "run_command": self._run_command,
            "read_artifact": self._read_artifact,
            "finish_task": self._finish_task,
        }
        if self.role == "implementer":
            methods["apply_patch"] = self._apply_patch
        method = methods.get(name)
        if method is None:
            raise HarnessError(f"tool is unavailable for {self.role}: {name}")
        if not isinstance(arguments, dict):
            raise HarnessError("tool arguments must be an object")
        temporary_context = self.tool_context is None
        if temporary_context:
            self.tool_context = {"tool": name}
        try:
            if self.role == "auditor":
                self._assert_auditor_unchanged()
            return method(arguments)
        finally:
            try:
                if self.role == "auditor":
                    self._assert_auditor_unchanged()
            finally:
                if temporary_context:
                    self.tool_context = None

    def _finish_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.role == "implementer":
            if arguments.get("status") != "READY_FOR_AUDIT":
                raise HarnessError("invalid implementer final status")
        elif arguments.get("verdict") not in {"APPROVE", "REJECT", "BLOCKED"}:
            raise HarnessError("invalid auditor final verdict")
        return {"ok": True, "contract": arguments}

    def workspace_fingerprint(self) -> str:
        digest = hashlib.sha256()
        head = self._process(
            [self.git_executable, "rev-parse", "--verify", "HEAD"],
            timeout_seconds=60,
            purpose="workspace-head",
        ).stdout
        index = self._process(
            [self.git_executable, "ls-files", "--stage", "-z"],
            timeout_seconds=60,
            purpose="workspace-index",
        ).stdout
        digest.update(b"head\0")
        digest.update(head)
        digest.update(b"index\0")
        digest.update(index)
        self._fingerprint_tree(digest, self.workspace, PurePosixPath("."), root=True)
        return digest.hexdigest()

    def _fingerprint_tree(
        self,
        digest: Any,
        directory: Path,
        relative: PurePosixPath,
        *,
        root: bool = False,
    ) -> None:
        try:
            before = directory.stat(follow_symlinks=False)
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda item: os.fsencode(item.name))
        except (FileNotFoundError, NotADirectoryError, PermissionError) as error:
            raise HarnessError(f"workspace changed or is unreadable: {directory}") from error
        for entry in entries:
            if root and entry.name == ".git":
                continue
            path = Path(entry.path)
            child = PurePosixPath(entry.name) if root else relative / entry.name
            raw_path = os.fsencode(child.as_posix())
            try:
                metadata = entry.stat(follow_symlinks=False)
            except (FileNotFoundError, PermissionError) as error:
                raise HarnessError(f"workspace changed or is unreadable: {path}") from error
            digest.update(b"path\0")
            digest.update(raw_path)
            digest.update(b"\0")
            digest.update(f"{metadata.st_mode:o}\0{metadata.st_size}\0{metadata.st_mtime_ns}\0".encode())
            if stat.S_ISLNK(metadata.st_mode):
                try:
                    target = os.readlink(path)
                except OSError as error:
                    raise HarnessError(f"workspace symlink changed while fingerprinting: {path}") from error
                digest.update(b"symlink\0")
                digest.update(os.fsencode(target))
            elif stat.S_ISDIR(metadata.st_mode):
                digest.update(b"directory\0")
                self._fingerprint_tree(digest, path, child)
            elif stat.S_ISREG(metadata.st_mode):
                digest.update(b"file\0")
                self._fingerprint_file(digest, path, metadata)
            else:
                digest.update(b"special\0")
        try:
            after = directory.stat(follow_symlinks=False)
        except (FileNotFoundError, PermissionError) as error:
            raise HarnessError(f"workspace changed while fingerprinting: {directory}") from error
        if self._stat_identity(before) != self._stat_identity(after):
            raise HarnessError(f"workspace changed while fingerprinting: {directory}")

    @staticmethod
    def _fingerprint_file(digest: Any, path: Path, expected: os.stat_result) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as error:
            raise HarnessError(f"workspace file changed or is unreadable: {path}") from error
        try:
            before = os.fstat(descriptor)
            if WorkspaceTools._stat_identity(before) != WorkspaceTools._stat_identity(expected):
                raise HarnessError(f"workspace file changed while fingerprinting: {path}")
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
            after = os.fstat(descriptor)
            if WorkspaceTools._stat_identity(before) != WorkspaceTools._stat_identity(after):
                raise HarnessError(f"workspace file changed while fingerprinting: {path}")
        finally:
            os.close(descriptor)

    @staticmethod
    def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def _assert_auditor_unchanged(self) -> None:
        expected = self.auditor_baseline_fingerprint
        if not isinstance(expected, str) or self.workspace_fingerprint() != expected:
            raise HarnessError("auditor workspace has tracked, untracked, or ignored changes")

    def _repo_path(self, value: Any, *, allow_root: bool = False) -> tuple[Path, str]:
        if not isinstance(value, str) or "\0" in value:
            raise HarnessError("path must be a string")
        if value in ("", ".") and allow_root:
            return self.workspace, "."
        candidate = PurePosixPath(value)
        if candidate.is_absolute() or ".." in candidate.parts or ".git" in candidate.parts:
            raise HarnessError(f"unsafe repository path: {value}")
        if any(PROTECTED_FILE_RE.fullmatch(part) for part in candidate.parts):
            raise HarnessError(f"protected repository path: {value}")
        normalized = candidate.as_posix()
        if normalized in ("", "."):
            raise HarnessError(f"empty repository path: {value}")
        logical = self.workspace
        for part in candidate.parts:
            logical = logical / part
            if logical.is_symlink():
                raise HarnessError(f"repository path traverses a symlink: {value}")
        target = logical.resolve(strict=False)
        try:
            resolved = target.relative_to(self.workspace)
        except ValueError as error:
            raise HarnessError(f"path escapes workspace: {value}") from error
        if any(PROTECTED_FILE_RE.fullmatch(part) for part in resolved.parts):
            raise HarnessError(f"protected repository path: {value}")
        return target, normalized

    def _read_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        target, normalized = self._repo_path(arguments.get("path"))
        if target.is_symlink() or not target.is_file():
            raise HarnessError(f"not a regular repository file: {normalized}")
        data = target.read_bytes()
        if len(data) > MAX_READ_BYTES or b"\0" in data:
            raise HarnessError(f"file is binary or exceeds {MAX_READ_BYTES} bytes: {normalized}")
        start = arguments.get("start_line", 1)
        count = arguments.get("line_count", 400)
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            raise HarnessError("start_line must be an integer >= 1")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 2000:
            raise HarnessError("line_count must be an integer from 1 to 2000")
        lines = data.decode("utf-8", errors="replace").splitlines()
        selected = lines[start - 1 : start - 1 + count]
        return {
            "ok": True,
            "path": normalized,
            "start_line": start,
            "end_line": start + len(selected) - 1,
            "total_lines": len(lines),
            "content": "\n".join(selected),
        }

    def _list_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        pattern = arguments.get("glob", "*")
        limit = arguments.get("limit", 400)
        if not isinstance(pattern, str) or "\0" in pattern:
            raise HarnessError("glob must be a string")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 1000:
            raise HarnessError("limit must be an integer from 1 to 1000")
        result = self._process(
            [self.git_executable, "ls-files", "--cached", "--others", "-z"],
            timeout_seconds=60,
        )
        values = result.stdout.split(b"\0")
        paths = []
        for raw in values:
            if not raw:
                continue
            path = raw.decode("utf-8", errors="surrogateescape")
            candidate = PurePosixPath(path)
            if ".git" in candidate.parts or any(PROTECTED_FILE_RE.fullmatch(p) for p in candidate.parts):
                continue
            if pattern not in ("", "*") and not fnmatch.fnmatchcase(path, pattern):
                continue
            paths.append(path)
        paths.sort()
        return {"ok": True, "files": paths[:limit], "truncated": len(paths) > limit}

    def _search_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query")
        if not isinstance(query, str) or not query or len(query) > 4096 or "\0" in query:
            raise HarnessError("query must be a non-empty bounded string")
        target, normalized = self._repo_path(arguments.get("path", "."), allow_root=True)
        pattern = arguments.get("glob")
        if pattern is not None and (not isinstance(pattern, str) or "\0" in pattern):
            raise HarnessError("glob must be a string")
        limit = arguments.get("limit", 200)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
            raise HarnessError("limit must be an integer from 1 to 500")
        fixed = arguments.get("fixed_strings", False)
        if not isinstance(fixed, bool):
            raise HarnessError("fixed_strings must be a boolean")
        expression = None
        if not fixed:
            try:
                expression = re.compile(query)
            except re.error as error:
                raise HarnessError(f"invalid search expression: {error}") from error
        if self.role == "auditor":
            python = trusted_executable("python3")
            helper_arguments = canonical_json(
                [
                    query,
                    normalized,
                    pattern,
                    fixed,
                    limit,
                    MAX_READ_BYTES,
                    MAX_SEARCH_BYTES,
                ]
            )
            command = " ".join(
                shlex.quote(value)
                for value in (python, "-I", "-c", SEARCH_HELPER_CODE, helper_arguments)
            )
            result = self._process(
                self._sandboxed_test_argv(command),
                timeout_seconds=min(self.command_timeout_seconds, 120),
                purpose="auditor-search",
            )
            try:
                payload = json.loads(result.stdout)
            except json.JSONDecodeError as error:
                raise HarnessError("sandboxed auditor search returned malformed output") from error
            if (
                not isinstance(payload, dict)
                or payload.get("ok") is not True
                or not isinstance(payload.get("matches"), list)
                or not isinstance(payload.get("truncated"), bool)
            ):
                raise HarnessError("sandboxed auditor search returned an invalid result")
            return payload
        pending = [target]
        matches = []
        scanned = 0
        truncated = False
        while pending and not truncated:
            current = pending.pop()
            if current.is_symlink():
                continue
            if current.is_dir():
                try:
                    with os.scandir(current) as stream:
                        entries = sorted(stream, key=lambda item: os.fsencode(item.name))
                except (FileNotFoundError, PermissionError) as error:
                    raise HarnessError(f"search path changed or is unreadable: {current}") from error
                for entry in reversed(entries):
                    candidate = Path(entry.path)
                    relative = candidate.relative_to(self.workspace)
                    if ".git" in relative.parts or any(
                        PROTECTED_FILE_RE.fullmatch(part) for part in relative.parts
                    ):
                        continue
                    pending.append(candidate)
                continue
            relative = current.relative_to(self.workspace).as_posix()
            if pattern and not fnmatch.fnmatchcase(relative, pattern):
                continue
            try:
                metadata = current.stat(follow_symlinks=False)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > MAX_READ_BYTES:
                    continue
                scanned += metadata.st_size
                if scanned > MAX_SEARCH_BYTES:
                    truncated = True
                    break
                data = current.read_bytes()
            except (FileNotFoundError, PermissionError) as error:
                raise HarnessError(f"search path changed or is unreadable: {current}") from error
            if b"\0" in data:
                continue
            for line_number, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                found = query in line if fixed else expression.search(line) is not None
                if not found:
                    continue
                matches.append(f"{relative}:{line_number}:{line[:4096]}")
                if len(matches) > limit:
                    truncated = True
                    break
        return {"ok": True, "matches": matches[:limit], "truncated": truncated}

    def _run_command(self, arguments: dict[str, Any]) -> dict[str, Any]:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip() or len(command) > 4096:
            raise HarnessError("command must be a non-empty bounded string")
        declared = set(self.task.get("test_commands", []))
        is_declared_test = command in declared
        if is_declared_test:
            argv = self._sandboxed_test_argv(command)
        else:
            argv = shlex.split(command)
            self._validate_read_only_command(argv)
            if argv[0] == "git":
                argv[0] = self.git_executable
            elif argv[0] == "pwd":
                argv[0] = self.pwd_executable
            if self.role == "auditor":
                argv = self._sandboxed_test_argv(
                    " ".join(shlex.quote(value) for value in argv)
                )
        result = self._process(argv, timeout_seconds=self.command_timeout_seconds, check=False)
        stdout, stdout_truncated = _bounded_text(result.stdout, MAX_COMMAND_BYTES)
        stderr, stderr_truncated = _bounded_text(result.stderr, MAX_COMMAND_BYTES)
        return {
            "ok": result.returncode == 0,
            "command": command,
            "exit_code": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": (
                stdout_truncated
                or stderr_truncated
                or result.stdout_truncated
                or result.stderr_truncated
            ),
            "declared_test": is_declared_test,
        }

    def _sandboxed_test_argv(self, command: str) -> list[str]:
        system = platform.system()
        shell = [trusted_executable("bash"), "--noprofile", "--norc", "-c", command]
        if system == "Darwin":
            try:
                executable = trusted_executable("sandbox-exec")
            except HarnessError as error:
                raise HarnessError("sandbox-exec is required for declared tests") from error
            profile = self._macos_sandbox_profile()
            profile_path = self.session_root / "declared-tests.sb"
            encoded = profile.encode("utf-8")
            current_profile = None
            if profile_path.exists():
                current_profile = _session_read_bytes(
                    self.session_root, Path("declared-tests.sb")
                )
            if current_profile != encoded:
                if self._tree_bytes(
                    self.session_root,
                    self.max_session_bytes + 1,
                ) + len(encoded) > self.max_session_bytes:
                    raise HarnessError("session artifact budget would be exceeded by sandbox profile")
                _session_atomic_bytes(
                    self.session_root,
                    Path("declared-tests.sb"),
                    encoded,
                )
            return [executable, "-f", str(profile_path), *shell]
        if system == "Linux":
            try:
                executable = trusted_executable("bwrap")
            except HarnessError as error:
                raise HarnessError("bubblewrap is required for declared tests") from error
            argv = [
                executable,
                "--die-with-parent",
                "--new-session",
                "--unshare-all",
                "--cap-drop",
                "ALL",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--dir",
                "/etc",
            ]
            for path in ("/usr", "/bin", "/sbin", "/lib", "/lib64"):
                if Path(path).exists():
                    argv.extend(("--ro-bind", path, path))
            for path in (
                "/etc/ld.so.cache",
                "/etc/nsswitch.conf",
                "/etc/passwd",
                "/etc/group",
                "/etc/localtime",
            ):
                if Path(path).exists():
                    argv.extend(("--ro-bind", path, path))
            argv.extend(
                (
                    "--ro-bind" if self.role == "auditor" else "--bind",
                    str(self.workspace),
                    "/workspace",
                    "--ro-bind",
                    str(self.workspace / ".git"),
                    "/workspace/.git",
                    "--bind",
                    str(self.sandbox_home),
                    "/sandbox-home",
                    "--bind",
                    str(self.sandbox_tmp),
                    "/sandbox-tmp",
                    "--clearenv",
                    "--setenv",
                    "HOME",
                    "/sandbox-home",
                    "--setenv",
                    "TMPDIR",
                    "/sandbox-tmp",
                    "--setenv",
                    "XDG_CONFIG_HOME",
                    "/sandbox-home/.config",
                    "--setenv",
                    "XDG_CACHE_HOME",
                    "/sandbox-home/.cache",
                    "--setenv",
                    "XDG_DATA_HOME",
                    "/sandbox-home/.local/share",
                    "--setenv",
                    "PATH",
                    "/usr/bin:/bin:/usr/sbin:/sbin",
                    "--setenv",
                    "LANG",
                    "C",
                    "--setenv",
                    "LC_ALL",
                    "C",
                    "--setenv",
                    "SHELL",
                    "/bin/bash",
                    "--setenv",
                    "USER",
                    "oxalpha",
                    "--setenv",
                    "LOGNAME",
                    "oxalpha",
                    "--setenv",
                    "PYTHONDONTWRITEBYTECODE",
                    "1",
                    "--setenv",
                    "GIT_CONFIG_NOSYSTEM",
                    "1",
                    "--setenv",
                    "GIT_CONFIG_GLOBAL",
                    "/dev/null",
                )
            )
            for protected in self._protected_workspace_paths():
                relative = protected.relative_to(self.workspace).as_posix()
                target = f"/workspace/{relative}"
                if protected.is_dir() and not protected.is_symlink():
                    argv.extend(("--tmpfs", target))
                else:
                    argv.extend(("--ro-bind", "/dev/null", target))
            argv.extend(("--chdir", "/workspace", *shell))
            return argv
        raise HarnessError(f"declared tests have no supported OS sandbox on {system}")

    def _macos_sandbox_profile(self) -> str:
        quote = lambda path: json.dumps(str(path), ensure_ascii=True)
        literals = [
            "/",
            "/Library",
            "/Library/Developer",
            "/dev/null",
            "/dev/random",
            "/dev/urandom",
        ]
        readable = [
            "/System/Library",
            "/System/Cryptexes",
            "/System/Volumes/Preboot/Cryptexes/OS",
            "/usr/bin",
            "/usr/lib",
            "/usr/libexec",
            "/usr/sbin",
            "/usr/share",
            "/bin",
            "/sbin",
            "/Library/Developer/CommandLineTools",
            "/Applications/Xcode.app",
            self.workspace,
            self.sandbox_home,
            self.sandbox_tmp,
        ]
        writable = [self.sandbox_home, self.sandbox_tmp]
        if self.role == "implementer":
            writable.insert(0, self.workspace)
        lines = [
            "(version 1)",
            "(deny default)",
            "(allow process*)",
            "(allow signal (target self))",
            "(allow file-read*",
            *[f"    (subpath {quote(path)} )" for path in readable],
            *[f"    (literal {quote(path)} )" for path in literals],
            ")",
            "(allow file-write*",
            *[f"    (subpath {quote(path)} )" for path in writable],
            '    (literal "/dev/null")',
            ")",
            f"(deny file-write* (subpath {quote(self.workspace / '.git')}))",
            *[
                f"(deny file-read* (subpath {quote(path)}))"
                for path in self._protected_workspace_paths()
                if path.is_dir() and not path.is_symlink()
            ],
            *[
                f"(deny file-read* (literal {quote(path)}))"
                for path in self._protected_workspace_paths()
                if not path.is_dir() or path.is_symlink()
            ],
            "(deny network*)",
            "",
        ]
        return "\n".join(lines)

    def _protected_workspace_paths(self) -> list[Path]:
        protected = []
        pending = [self.workspace]
        while pending:
            current = pending.pop()
            try:
                with os.scandir(current) as stream:
                    entries = list(stream)
            except (FileNotFoundError, PermissionError) as error:
                raise HarnessError(f"cannot inspect workspace credentials: {current}") from error
            for entry in entries:
                if current == self.workspace and entry.name == ".git":
                    continue
                path = Path(entry.path)
                if PROTECTED_FILE_RE.fullmatch(entry.name):
                    protected.append(path)
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(path)
        return sorted(protected, key=lambda path: os.fsencode(str(path)))

    def _read_artifact(self, arguments: dict[str, Any]) -> dict[str, Any]:
        artifact = arguments.get("artifact")
        if not isinstance(artifact, str) or not re.fullmatch(r"tool-results/[0-9A-Za-z_.-]+\.json", artifact):
            raise HarnessError("invalid session artifact name")
        offset = arguments.get("offset", 0)
        maximum = arguments.get("max_bytes", 32768)
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            raise HarnessError("offset must be an integer >= 0")
        if not isinstance(maximum, int) or isinstance(maximum, bool) or not 1 <= maximum <= 65536:
            raise HarnessError("max_bytes must be an integer from 1 to 65536")
        descriptor, name = _session_parent_fd(self.session_root, Path(artifact))
        child = -1
        try:
            _reject_existing_symlink(descriptor, name)
            child = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            metadata = os.fstat(child)
            if not stat.S_ISREG(metadata.st_mode):
                raise HarnessError("artifact is not a regular file")
            os.lseek(child, offset, os.SEEK_SET)
            chunk = os.read(child, maximum)
            more = os.read(child, 1) != b""
        except FileNotFoundError as error:
            raise HarnessError("artifact does not exist") from error
        finally:
            if child >= 0:
                os.close(child)
            os.close(descriptor)
        return {
            "ok": True,
            "artifact": artifact,
            "offset": offset,
            "next_offset": offset + len(chunk),
            "more": more,
            "content": chunk.decode("utf-8", errors="replace"),
        }

    def _apply_patch(self, arguments: dict[str, Any]) -> dict[str, Any]:
        patch = arguments.get("patch")
        if not isinstance(patch, str) or not patch:
            raise HarnessError("patch must be a non-empty string")
        encoded = patch.encode("utf-8")
        if len(encoded) > MAX_PATCH_BYTES or b"\0" in encoded:
            raise HarnessError("patch is binary or too large")
        if "GIT binary patch" in patch or "new file mode 120000" in patch or "new file mode 160000" in patch:
            raise HarnessError("binary, symlink, and submodule patches are forbidden")
        paths = self._patch_paths(encoded)
        outside = sorted(path for path in paths if not path_allowed(path, self.task.get("write_set", [])))
        if outside:
            raise HarnessError(f"patch paths outside declared write_set: {outside}")
        for path in paths:
            self._repo_path(path)
        check = self._process(
            [self.git_executable, "apply", "--check", "--whitespace=nowarn", "-"],
            input_bytes=encoded,
            timeout_seconds=60,
            check=False,
        )
        if check.returncode != 0:
            raise HarnessError(check.stderr.decode("utf-8", errors="replace")[-4000:])
        applied = self._process(
            [self.git_executable, "apply", "--whitespace=nowarn", "-"],
            input_bytes=encoded,
            timeout_seconds=60,
            check=False,
        )
        if applied.returncode != 0:
            raise HarnessError(applied.stderr.decode("utf-8", errors="replace")[-4000:])
        return {"ok": True, "changed_paths": sorted(paths)}

    def _patch_paths(self, patch: bytes) -> set[str]:
        text = patch.decode("utf-8")
        for marker in (
            "rename from ",
            "rename to ",
            "copy from ",
            "copy to ",
            "new file mode 120000",
            "new file mode 160000",
        ):
            if any(line.startswith(marker) for line in text.splitlines()):
                raise HarnessError("rename, copy, symlink, and submodule patches are forbidden")
        summary = self._process(
            [self.git_executable, "apply", "--numstat", "-z", "-"],
            input_bytes=patch,
            timeout_seconds=60,
            check=False,
        )
        if summary.returncode != 0:
            raise HarnessError(summary.stderr.decode("utf-8", errors="replace")[-4000:])
        paths = set()
        for record in summary.stdout.split(b"\0"):
            if not record:
                continue
            try:
                _added, _deleted, raw_path = record.split(b"\t", 2)
                path = raw_path.decode("utf-8", errors="strict")
            except (UnicodeDecodeError, ValueError) as error:
                raise HarnessError("malformed patch path summary") from error
            candidate = PurePosixPath(path)
            if candidate.is_absolute() or ".." in candidate.parts or ".git" in candidate.parts:
                raise HarnessError(f"unsafe patch path: {path}")
            normalized = candidate.as_posix()
            if any(PROTECTED_FILE_RE.fullmatch(part) for part in candidate.parts):
                raise HarnessError(f"protected patch path: {path}")
            paths.add(normalized)
        if not paths:
            raise HarnessError("patch changes no text files")
        return paths

    @staticmethod
    def _validate_read_only_command(argv: list[str]) -> None:
        if not argv:
            raise HarnessError("empty command")
        allowed = {
            ("pwd",),
            ("git", "status"),
            ("git", "status", "--short"),
            ("git", "status", "--porcelain"),
            ("git", "diff"),
            ("git", "diff", "--check"),
            ("git", "diff", "--stat"),
            ("git", "diff", "--name-only"),
            ("git", "ls-files"),
            ("git", "rev-parse", "HEAD"),
        }
        if tuple(argv) not in allowed:
            raise HarnessError("command is not a declared test or approved read-only inspection")

    @staticmethod
    def _darwin_libproc() -> Any:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pidinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        library.proc_pidinfo.restype = ctypes.c_int
        library.proc_pidpath.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_uint32,
        )
        library.proc_pidpath.restype = ctypes.c_int
        library.proc_pidfdinfo.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        library.proc_pidfdinfo.restype = ctypes.c_int
        library.proc_listchildpids.argtypes = (
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        library.proc_listchildpids.restype = ctypes.c_int
        library.proc_listpids.argtypes = (
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        )
        library.proc_listpids.restype = ctypes.c_int
        return library

    @staticmethod
    def _darwin_libsandbox() -> Any:
        try:
            library = ctypes.CDLL("/usr/lib/libsandbox.dylib", use_errno=True)
        except OSError as error:
            raise HarnessError("macOS process containment API is unavailable") from error
        library.sandbox_check.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        library.sandbox_check.restype = ctypes.c_int
        return library

    @staticmethod
    def _darwin_sandbox_decision(pid: int, path: Path) -> int | None:
        library = WorkspaceTools._darwin_libsandbox()
        ctypes.set_errno(0)
        result = library.sandbox_check(
            pid,
            b"file-read-data",
            1,
            ctypes.c_char_p(os.fsencode(path)),
        )
        if result >= 0:
            return result
        saved_errno = ctypes.get_errno()
        if saved_errno == errno.ESRCH:
            return None
        raise HarnessError("cannot inspect macOS process containment")

    @staticmethod
    def _process_identity(pid: int) -> dict[str, Any] | None:
        system = platform.system()
        if system == "Linux":
            root = Path("/proc") / str(pid)
            try:
                raw = (root / "stat").read_bytes()
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                executable = os.readlink(root / "exe")
                cwd = os.readlink(root / "cwd")
                uid = (root / "status").stat().st_uid
            except FileNotFoundError:
                return None
            except (OSError, PermissionError) as error:
                raise HarnessError(f"cannot inspect process identity for pid {pid}") from error
            closing = raw.rfind(b")")
            if closing < 0:
                raise HarnessError(f"malformed process identity for pid {pid}")
            fields = raw[closing + 2 :].split()
            if len(fields) < 20:
                raise HarnessError(f"short process identity for pid {pid}")
            return {
                "platform": "Linux",
                "pid": pid,
                "ppid": int(fields[1]),
                "pgid": int(fields[2]),
                "uid": uid,
                "birth": f"{boot_id}:{fields[19].decode()}",
                "executable": os.path.realpath(executable),
                "cwd": os.path.realpath(cwd),
            }
        if system == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            info = _DarwinProcBsdInfo()
            size = ctypes.sizeof(info)
            result = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
            if result == 0:
                return None
            if result != size:
                raise HarnessError(f"short process identity for pid {pid}")
            path = ctypes.create_string_buffer(4096)
            length = library.proc_pidpath(pid, path, len(path))
            if length <= 0:
                latest = _DarwinProcBsdInfo()
                latest_result = library.proc_pidinfo(
                    pid,
                    3,
                    0,
                    ctypes.byref(latest),
                    size,
                )
                if latest_result != size or latest.pbi_status == 5:
                    return None
                raise HarnessError(f"cannot inspect process executable for pid {pid}")
            executable = os.fsdecode(path.value)
            paths = _DarwinProcVnodePathInfo()
            path_size = ctypes.sizeof(paths)
            path_result = library.proc_pidinfo(pid, 9, 0, ctypes.byref(paths), path_size)
            if path_result != path_size:
                latest = _DarwinProcBsdInfo()
                latest_result = library.proc_pidinfo(
                    pid,
                    3,
                    0,
                    ctypes.byref(latest),
                    size,
                )
                if latest_result != size or latest.pbi_status == 5:
                    return None
                raise HarnessError(f"cannot inspect process workspace for pid {pid}")
            cwd = os.fsdecode(paths.pvi_cdir.vip_path)
            return {
                "platform": "Darwin",
                "pid": pid,
                "ppid": int(info.pbi_ppid),
                "pgid": int(info.pbi_pgid),
                "uid": int(info.pbi_uid),
                "birth": f"{info.pbi_start_tvsec}:{info.pbi_start_tvusec}",
                "executable": os.path.realpath(executable),
                "cwd": os.path.realpath(cwd),
            }
        raise HarnessError(f"process identity is unsupported on {system}")

    def _identity_matches(self, record: dict[str, Any], identity: dict[str, Any]) -> bool:
        expected = record.get("birth_identity")
        executables = record.get("allowed_executables")
        matches = (
            record.get("workspace") == str(self.workspace)
            and record.get("session_root") == str(self.session_root)
            and record.get("session_id") == self.session_root.name
            and isinstance(expected, dict)
            and identity.get("pid") == expected.get("pid")
            and identity.get("birth") == expected.get("birth")
            and identity.get("uid") == expected.get("uid") == os.getuid()
            and isinstance(executables, list)
            and expected.get("executable") in executables
            and self._record_session_marker_matches(record)
        )
        if not matches:
            return False
        if record.get("containment") == "darwin_sandbox":
            try:
                return self._darwin_boundary_matches(identity["pid"], record)
            except HarnessError:
                return False
        return record.get("containment") in {
            "fork_disabled",
            "pid_namespace",
            "tracked_descendants",
        }

    @staticmethod
    def _birth_key(value: Any) -> tuple[str, int] | None:
        if not isinstance(value, str):
            return None
        fields = value.rsplit(":", 1)
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            return None
        return (fields[0], int(fields[1]))

    @staticmethod
    def _process_birth_identity(pid: int) -> tuple[str, int] | None:
        if platform.system() == "Linux":
            root = Path("/proc") / str(pid)
            try:
                raw = (root / "stat").read_bytes()
                boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
                uid = (root / "status").stat().st_uid
            except (FileNotFoundError, PermissionError):
                return None
            closing = raw.rfind(b")")
            fields = raw[closing + 2 :].split() if closing >= 0 else []
            if len(fields) < 20:
                return None
            return (f"{boot_id}:{fields[19].decode()}", uid)
        if platform.system() == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            info = _DarwinProcBsdInfo()
            size = ctypes.sizeof(info)
            result = library.proc_pidinfo(pid, 3, 0, ctypes.byref(info), size)
            if result != size or info.pbi_status == 5:
                return None
            return (
                f"{info.pbi_start_tvsec}:{info.pbi_start_tvusec}",
                int(info.pbi_uid),
            )
        raise HarnessError("process birth inspection is unsupported")

    @staticmethod
    def _darwin_vnode_identity(pid: int, descriptor: int) -> tuple[int, int] | None:
        library = WorkspaceTools._darwin_libproc()
        value = _DarwinVnodeFdInfo()
        size = ctypes.sizeof(value)
        result = library.proc_pidfdinfo(pid, descriptor, 1, ctypes.byref(value), size)
        if result == 0:
            return None
        if result != size:
            raise HarnessError(f"short vnode identity for pid {pid} fd {descriptor}")
        return (int(value.vnode.vi_stat.vst_dev), int(value.vnode.vi_stat.vst_ino))

    @staticmethod
    def _containment_marker(descriptor: int) -> dict[str, Any]:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise HarnessError("containment marker is not a regular file")
        if platform.system() == "Linux":
            identity = (metadata.st_dev, metadata.st_ino)
        if platform.system() == "Darwin":
            identity = WorkspaceTools._darwin_vnode_identity(os.getpid(), descriptor)
            if identity is None:
                raise HarnessError("cannot bind the containment marker vnode")
        if platform.system() not in {"Darwin", "Linux"}:
            raise HarnessError("process containment marker is unsupported")
        return {
            "platform": platform.system(),
            "kind": "vnode",
            "device": identity[0],
            "inode": identity[1],
        }

    def _record_session_marker_matches(self, record: dict[str, Any]) -> bool:
        marker = record.get("containment_marker")
        if not isinstance(marker, dict) or marker.get("kind") != "vnode":
            return False
        raw_path = marker.get("path")
        if not isinstance(raw_path, str):
            return False
        relative = PurePosixPath(raw_path)
        if (
            relative.is_absolute()
            or len(relative.parts) != 2
            or relative.parts[0] != "processes"
            or ".." in relative.parts
        ):
            return False
        descriptor = -1
        try:
            descriptor = _session_open_regular_nofollow(
                self.session_root,
                Path(relative.as_posix()),
            )
            current = self._containment_marker(descriptor)
        except (HarnessError, OSError):
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        return all(current.get(key) == marker.get(key) for key in ("platform", "kind", "device", "inode"))

    def _darwin_boundary_paths(self, record: dict[str, Any]) -> tuple[Path, Path]:
        boundary = record.get("os_boundary")
        if not isinstance(boundary, dict) or set(boundary) != {
            "schema_version",
            "kind",
            "denied_canary",
            "allowed_canary",
            "profile",
        }:
            raise HarnessError("process ownership record has no exact macOS boundary")
        if boundary.get("schema_version") != 1 or boundary.get("kind") != "sandbox-decision-pair":
            raise HarnessError("process ownership record has an invalid macOS boundary")
        paths = []
        for key, parent in (
            ("denied_canary", "sandbox-tmp"),
            ("allowed_canary", "sandbox-tmp"),
            ("profile", "processes"),
        ):
            raw = boundary.get(key)
            if not isinstance(raw, str):
                raise HarnessError("process ownership record has an invalid macOS boundary path")
            relative = PurePosixPath(raw)
            if (
                relative.is_absolute()
                or len(relative.parts) != 2
                or relative.parts[0] != parent
                or ".." in relative.parts
            ):
                raise HarnessError("process ownership record has an unsafe macOS boundary path")
            descriptor = _session_open_regular_nofollow(
                self.session_root,
                Path(relative.as_posix()),
            )
            os.close(descriptor)
            paths.append(self.session_root / relative.as_posix())
        if paths[0] == paths[1]:
            raise HarnessError("macOS containment canaries are not distinct")
        return paths[0], paths[1]

    def _darwin_boundary_matches(self, pid: int, record: dict[str, Any]) -> bool:
        denied, allowed = self._darwin_boundary_paths(record)
        denied_result = self._darwin_sandbox_decision(pid, denied)
        if denied_result is None:
            return False
        allowed_result = self._darwin_sandbox_decision(pid, allowed)
        if allowed_result is None:
            return False
        return denied_result > 0 and allowed_result == 0

    @staticmethod
    def _nested_codex_fallback_enabled() -> bool:
        return (
            platform.system() == "Darwin"
            and os.environ.get("CODEX_CI") == "1"
            and os.environ.get("CODEX_SANDBOX") == "seatbelt"
            and os.environ.get("OXALPHA_REAL_SANDBOX") != "1"
        )

    def _nested_fallback_target(self, target: list[str]) -> list[str]:
        sandbox = trusted_executable("sandbox-exec")
        if len(target) >= 4 and target[:2] == [sandbox, "-f"]:
            profile = self.session_root / "declared-tests.sb"
            if Path(target[2]) != profile:
                raise HarnessError("nested fallback rejected an unknown sandbox profile")
            target = target[3:]
        shell = trusted_executable("bash")
        if len(target) == 5 and target[:4] == [shell, "--noprofile", "--norc", "-c"]:
            try:
                target = shlex.split(target[4])
            except ValueError as error:
                raise HarnessError("nested fallback rejected malformed shell syntax") from error
            if target and target[0] == "git":
                target[0] = self.git_executable
            elif target and target[0] == "pwd":
                target[0] = self.pwd_executable
        if target == [self.pwd_executable]:
            return target
        if target and target[0] == self.python_executable:
            if len(target) == 5 and target[1:4] == ["-I", "-c", SEARCH_HELPER_CODE]:
                return target
            raise HarnessError("nested fallback rejected non-controller Python code")
        if not target or target[0] != self.git_executable:
            raise HarnessError("nested fallback rejected a non-allowlisted executable")
        arguments = tuple(target[1:])
        read_only = {
            ("rev-parse", "--verify", "HEAD"),
            ("rev-parse", "HEAD"),
            ("ls-files", "--stage", "-z"),
            ("ls-files", "--cached", "--others", "-z"),
            ("ls-files",),
            ("status",),
            ("status", "--short"),
            ("status", "--porcelain"),
            ("diff",),
            ("diff", "--check"),
            ("diff", "--stat"),
            ("diff", "--name-only"),
        }
        patch_operations = {
            ("apply", "--check", "--whitespace=nowarn", "-"),
            ("apply", "--whitespace=nowarn", "-"),
            ("apply", "--numstat", "-z", "-"),
        }
        if arguments in patch_operations:
            if self.role != "implementer" or (self.tool_context or {}).get("tool") != "apply_patch":
                raise HarnessError("nested fallback rejected an unowned mutating git command")
        elif arguments not in read_only:
            raise HarnessError("nested fallback rejected non-allowlisted git arguments")
        safe = [self.git_executable, "-c", "core.fsmonitor=false"]
        if arguments and arguments[0] == "diff":
            return [*safe, "diff", "--no-ext-diff", *arguments[1:]]
        return [*safe, *arguments]

    def _prepare_os_containment(
        self,
        process_id: str,
        denied_relative: Path,
        allowed_relative: Path | None,
        launch_argv: list[str],
        *,
        linux_pid_namespace: bool = False,
    ) -> tuple[list[str], str, dict[str, Any] | None]:
        system = platform.system()
        if system == "Linux":
            containment = "pid_namespace" if linux_pid_namespace else "tracked_descendants"
            return launch_argv, containment, None
        if system != "Darwin":
            raise HarnessError(f"process containment is unsupported on {system}")
        if allowed_relative is None:
            raise HarnessError("macOS process containment has no control canary")
        if self._nested_codex_fallback_enabled():
            if (
                len(launch_argv) < 9
                or launch_argv[:4] != [
                    self.python_executable,
                    "-I",
                    "-c",
                    PROCESS_GATE_CODE,
                ]
            ):
                raise HarnessError("nested fallback rejected an unknown process gate")
            target = self._nested_fallback_target(launch_argv[8:])
            fallback = [*launch_argv[:8], *target]
            fallback[7] = "1"
            return fallback, "fork_disabled", None
        try:
            executable = trusted_executable("sandbox-exec")
            self._darwin_libsandbox()
        except HarnessError as error:
            raise HarnessError("macOS process containment is unavailable") from error
        denied = self.session_root / denied_relative
        allowed = self.session_root / allowed_relative
        if (
            self._darwin_sandbox_decision(os.getpid(), denied) != 0
            or self._darwin_sandbox_decision(os.getpid(), allowed) != 0
        ):
            raise HarnessError("macOS process containment canaries are not uniquely inspectable")
        profile_relative = Path("processes") / f"{process_id}.sb"
        profile_path = self.session_root / profile_relative
        quote = lambda value: json.dumps(str(value), ensure_ascii=True)
        profile = "\n".join(
            (
                "(version 1)",
                "(allow default)",
                f"(deny file-read-data (literal {quote(denied)}))",
                "(deny file-write*",
                f"    (literal {quote(denied)})",
                f"    (literal {quote(allowed)})",
                f"    (literal {quote(profile_path)})",
                ")",
                "",
            )
        )
        session_bytes, session_entries = self._tree_usage(
            self.session_root,
            self.max_session_bytes + 1,
            self.max_storage_entries + 1,
        )
        if (
            session_bytes + len(profile.encode("utf-8")) > self.max_session_bytes
            or session_entries + 1 > self.max_storage_entries
        ):
            raise HarnessError("session storage budget would be exceeded by process containment")
        _session_exclusive_bytes(self.session_root, profile_relative, profile.encode("utf-8"))
        boundary = {
            "schema_version": 1,
            "kind": "sandbox-decision-pair",
            "denied_canary": denied_relative.as_posix(),
            "allowed_canary": allowed_relative.as_posix(),
            "profile": profile_relative.as_posix(),
        }
        return [executable, "-f", str(profile_path), *launch_argv], "darwin_sandbox", boundary

    def _live_containment_matches(self, pid: int, record: dict[str, Any]) -> bool:
        if record.get("containment") == "darwin_sandbox":
            return self._darwin_boundary_matches(pid, record)
        marker = record.get("containment_marker")
        if not isinstance(marker, dict):
            return False
        return self._process_has_marker(pid, marker)

    @staticmethod
    def _process_has_marker(pid: int, marker: dict[str, Any]) -> bool:
        if marker.get("kind") != "vnode" or marker.get("platform") != platform.system():
            raise HarnessError("process ownership record has an invalid containment marker")
        expected = (marker.get("device"), marker.get("inode"))
        if not all(isinstance(value, int) for value in expected):
            raise HarnessError("process ownership record has an invalid vnode identity")
        if platform.system() == "Linux":
            try:
                entries = list((Path("/proc") / str(pid) / "fd").iterdir())
            except FileNotFoundError:
                return False
            except PermissionError as error:
                raise HarnessError(f"cannot inspect process descriptors for pid {pid}") from error
            for path in entries:
                try:
                    metadata = path.stat()
                except (FileNotFoundError, PermissionError):
                    continue
                if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == expected:
                    return True
            return False
        if platform.system() == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            item_size = ctypes.sizeof(_DarwinProcFdInfo)
            size = library.proc_pidinfo(pid, 1, 0, None, 0)
            if size <= 0:
                return False
            values = (_DarwinProcFdInfo * (size // item_size + 16))()
            used = library.proc_pidinfo(pid, 1, 0, values, ctypes.sizeof(values))
            if used <= 0:
                return False
            for value in values[: used // item_size]:
                if value.proc_fdtype != 1:
                    continue
                identity = WorkspaceTools._darwin_vnode_identity(pid, value.proc_fd)
                if identity == expected:
                    return True
            return False
        raise HarnessError("process session marker inspection is unsupported")

    @staticmethod
    def _user_process_ids() -> list[int]:
        if platform.system() == "Linux":
            values = []
            for path in Path("/proc").glob("[0-9]*/status"):
                try:
                    uid = next(
                        int(line.split()[1])
                        for line in path.read_text().splitlines()
                        if line.startswith("Uid:")
                    )
                except (FileNotFoundError, PermissionError, StopIteration, ValueError):
                    continue
                if uid == os.getuid():
                    values.append(int(path.parent.name))
            return values
        if platform.system() == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            size = library.proc_listpids(4, os.getuid(), None, 0)
            if size <= 0:
                raise HarnessError("cannot list processes for containment")
            values = (ctypes.c_int * (size // ctypes.sizeof(ctypes.c_int) + 16))()
            used = library.proc_listpids(4, os.getuid(), values, ctypes.sizeof(values))
            if used < 0:
                raise HarnessError("cannot list processes for containment")
            return [
                int(value)
                for value in values[: used // ctypes.sizeof(ctypes.c_int)]
                if value > 0
            ]
        raise HarnessError("process listing is unsupported")

    def _marked_processes(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        marker = record.get("containment_marker")
        if not isinstance(marker, dict):
            raise HarnessError("process ownership record has no valid session marker")
        uses_darwin_boundary = record.get("containment") == "darwin_sandbox"
        if uses_darwin_boundary:
            self._darwin_boundary_paths(record)
        root_birth = self._birth_key(record.get("birth_identity", {}).get("birth"))
        if root_birth is None:
            raise HarnessError("process ownership record has no valid root birth identity")
        matches = []
        for pid in self._user_process_ids():
            brief = self._process_birth_identity(pid)
            if brief is None or brief[1] != os.getuid():
                continue
            birth = self._birth_key(brief[0])
            if birth is None or birth < root_birth:
                continue
            if uses_darwin_boundary:
                if not self._darwin_boundary_matches(pid, record):
                    continue
            elif not self._process_has_marker(pid, marker):
                continue
            identity = self._process_identity(pid)
            if identity is None:
                continue
            if identity.get("uid") != os.getuid():
                raise HarnessError(f"process session marker changed owner for pid {pid}")
            matches.append(identity)
        return matches

    def _refresh_root_identity(
        self,
        record: dict[str, Any],
        record_path: Path,
    ) -> dict[str, Any] | None:
        pid = record.get("pid")
        if not isinstance(pid, int):
            return None
        current = self._process_identity(pid)
        if current is None:
            return None
        expected = record.get("birth_identity")
        if (
            not isinstance(expected, dict)
            or current.get("pid") != expected.get("pid")
            or current.get("birth") != expected.get("birth")
            or current.get("uid") != expected.get("uid")
            or expected.get("uid") != os.getuid()
            or not self._record_session_marker_matches(record)
        ):
            raise HarnessError(f"owned process birth identity changed for pid {pid}")
        if record.get("containment") == "darwin_sandbox" and not self._darwin_boundary_matches(
            pid,
            record,
        ):
            raise HarnessError(f"owned process containment changed for pid {pid}")
        allowed = record.get("allowed_executables")
        if not isinstance(allowed, list):
            raise HarnessError("process ownership record has no executable binding")
        executable = current.get("executable")
        if not isinstance(executable, str):
            raise HarnessError(f"owned process has no executable identity for pid {pid}")
        observed_cwd = current.get("cwd")
        if executable not in allowed or record.get("observed_cwd") != observed_cwd:
            record["allowed_executables"] = sorted({*allowed, executable})
            record["current_executable"] = executable
            record["observed_cwd"] = observed_cwd
            self._write_process_record(record_path, record)
        return current

    @staticmethod
    def _child_pids(pid: int) -> list[int]:
        system = platform.system()
        if system == "Linux":
            children = []
            for stat_path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    raw = stat_path.read_bytes()
                except (FileNotFoundError, PermissionError):
                    continue
                closing = raw.rfind(b")")
                if closing < 0:
                    continue
                fields = raw[closing + 2 :].split()
                if len(fields) > 1 and int(fields[1]) == pid:
                    children.append(int(stat_path.parent.name))
            return children
        if system == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            capacity = MAX_PROCESS_DESCENDANTS + 2
            values = (ctypes.c_int * capacity)()
            count = library.proc_listchildpids(pid, values, ctypes.sizeof(values))
            if count < 0:
                raise HarnessError(f"cannot list child processes for pid {pid}")
            return [int(value) for value in values[: min(count, capacity)] if value > 0]
        raise HarnessError(f"descendant discovery is unsupported on {system}")

    @staticmethod
    def _user_process_count() -> int:
        system = platform.system()
        if system == "Linux":
            count = 0
            for status in Path("/proc").glob("[0-9]*/status"):
                try:
                    for line in status.read_text().splitlines():
                        if line.startswith("Uid:") and int(line.split()[1]) == os.getuid():
                            count += 1
                            break
                except (FileNotFoundError, PermissionError, ValueError):
                    continue
            return count
        if system == "Darwin":
            library = WorkspaceTools._darwin_libproc()
            size = library.proc_listpids(4, os.getuid(), None, 0)
            if size < 0:
                raise HarnessError("cannot count user processes")
            return max(1, size // ctypes.sizeof(ctypes.c_int))
        raise HarnessError(f"process counting is unsupported on {system}")

    def _discover_descendants(
        self,
        root_pid: int,
        record: dict[str, Any],
        *,
        discover_detached: bool = False,
    ) -> list[dict[str, Any]]:
        known = {
            int(item["pid"]): item
            for item in record.get("descendants", [])
            if isinstance(item, dict) and isinstance(item.get("pid"), int)
        }
        for pid, expected in list(known.items()):
            current = self._process_identity(pid)
            if current is None:
                continue
            if (
                current.get("birth") == expected.get("birth")
                and current.get("uid") == expected.get("uid")
            ):
                known[pid] = current
        pending = [root_pid, *known]
        visited = set()
        while pending:
            parent = pending.pop()
            if parent in visited:
                continue
            visited.add(parent)
            for child in self._child_pids(parent):
                if child == root_pid or child in known:
                    continue
                identity = self._process_identity(child)
                if identity is None:
                    continue
                known[child] = identity
                pending.append(child)
        if discover_detached:
            for identity in self._marked_processes(record):
                pid = identity["pid"]
                if pid != root_pid:
                    known[pid] = identity
        rendered = [known[pid] for pid in sorted(known)]
        if len(known) > self.max_process_descendants:
            raise HarnessError(
                f"process descendant limit of {self.max_process_descendants} was exceeded"
            )
        return rendered

    def _track_descendants(
        self,
        root_pid: int,
        record: dict[str, Any],
        record_path: Path,
        *,
        discover_detached: bool = False,
    ) -> None:
        rendered = self._discover_descendants(
            root_pid,
            record,
            discover_detached=discover_detached,
        )
        if rendered != record.get("descendants", []):
            record["descendants"] = rendered
            self._write_process_record(record_path, record)

    def _waited_fast_process_result(
        self,
        process: subprocess.Popen[bytes],
        argv: Sequence[str],
    ) -> ProcessResult | None:
        if process.poll() is None:
            return None
        if process.stdin is not None:
            process.stdin.close()
        streams = {
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        if any(stream is None for stream in streams.values()):
            raise HarnessError("completed child has no controller-owned output pipes")
        selector = selectors.DefaultSelector()
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        try:
            for name, stream in streams.items():
                selector.register(stream, selectors.EVENT_READ, name)
            deadline = time.monotonic() + 1
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HarnessError("completed child retained live output pipes")
                for key, _mask in selector.select(timeout=min(0.05, remaining)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    name = key.data
                    room = MAX_COMMAND_BYTES - len(buffers[name])
                    if room > 0:
                        buffers[name].extend(chunk[:room])
                    if len(chunk) > room:
                        truncated[name] = True
                if self._storage_budget_exceeded():
                    raise HarnessError(
                        f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
                    )
        finally:
            selector.close()
        process.wait(timeout=0)
        return ProcessResult(
            list(argv),
            process.returncode,
            bytes(buffers["stdout"]),
            bytes(buffers["stderr"]),
            truncated["stdout"],
            truncated["stderr"],
        )

    def _process(
        self,
        argv: Sequence[str],
        *,
        input_bytes: bytes | None = None,
        timeout_seconds: int,
        check: bool = True,
        purpose: str = "command",
    ) -> ProcessResult:
        if not argv or not Path(argv[0]).is_absolute():
            raise HarnessError("command executable must use an absolute trusted path")
        if self._storage_budget_exceeded():
            raise HarnessError(
                f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
            )
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C",
            "LC_ALL": "C",
            "SHELL": "/bin/bash",
            "USER": "oxalpha",
            "LOGNAME": "oxalpha",
            "HOME": str(self.sandbox_home),
            "TMPDIR": str(self.sandbox_tmp),
            "XDG_CONFIG_HOME": str(self.sandbox_home / ".config"),
            "XDG_CACHE_HOME": str(self.sandbox_home / ".cache"),
            "XDG_DATA_HOME": str(self.sandbox_home / ".local" / "share"),
            "PYTHONDONTWRITEBYTECODE": "1",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
        if platform.system() == "Darwin":
            environment["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"
        process_id = secrets.token_hex(16)
        process_token = secrets.token_hex(32)
        environment["OXALPHA_PROCESS_TOKEN"] = process_token
        marker_relative = Path("processes") / f"{process_id}.marker"
        denied_relative = Path("sandbox-tmp") / f".oxalpha-{process_id}.deny"
        allowed_relative = (
            Path("sandbox-tmp") / f".oxalpha-{process_id}.allow"
            if platform.system() == "Darwin"
            else None
        )
        session_bytes, session_entries = self._tree_usage(
            self.session_root,
            self.max_session_bytes + 1,
            self.max_storage_entries + 1,
        )
        reserved_entries = 5 if allowed_relative is not None else 2
        if (
            session_bytes + len(process_token) + 32 > self.max_session_bytes
            or session_entries + reserved_entries > self.max_storage_entries
        ):
            raise HarnessError("session storage budget would be exceeded by process ownership")
        _session_exclusive_bytes(self.session_root, marker_relative, process_token.encode())
        if allowed_relative is not None:
            _session_exclusive_bytes(self.session_root, denied_relative, b"denied\n")
            _session_exclusive_bytes(self.session_root, allowed_relative, b"allowed\n")
        containment_descriptor = _session_open_read_fd(self.session_root, marker_relative)
        try:
            containment_marker = self._containment_marker(containment_descriptor)
            containment_marker["path"] = marker_relative.as_posix()
        except BaseException:
            os.close(containment_descriptor)
            raise
        command_argv = list(argv)
        linux_pid_namespace = (
            platform.system() == "Linux" and Path(command_argv[0]).name == "bwrap"
        )
        if linux_pid_namespace:
            command_argv[1:1] = ["--keep-fd", str(containment_descriptor)]
        gate_read, gate_write = os.pipe()
        ready_read, ready_write = os.pipe()
        os.set_blocking(ready_read, False)
        try:
            process_limit = self._user_process_count() + self.max_process_descendants + 2
            gated_argv = [
                self.python_executable,
                "-I",
                "-c",
                PROCESS_GATE_CODE,
                str(gate_read),
                str(ready_write),
                str(self.max_session_bytes),
                str(process_limit),
                *command_argv,
            ]
            launch_argv, containment, boundary = self._prepare_os_containment(
                process_id,
                denied_relative,
                allowed_relative,
                gated_argv,
                linux_pid_namespace=linux_pid_namespace,
            )
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            os.close(ready_read)
            os.close(ready_write)
            os.close(containment_descriptor)
            raise
        record_path = self.process_root / f"{process_id}.json"
        record = {
            "schema_version": 1,
            "process_id": process_id,
            "process_token": process_token,
            "containment_marker": containment_marker,
            "workspace": str(self.workspace),
            "session_root": str(self.session_root),
            "session_id": self.session_root.name,
            "owner": dict(self.tool_context or {"purpose": purpose}),
            "purpose": purpose,
            "argv_sha256": sha256_bytes(canonical_json(launch_argv).encode()),
            "allowed_executables": sorted(
                {
                    os.path.realpath(self.python_executable),
                    os.path.realpath(str(argv[0])),
                    os.path.realpath(str(launch_argv[0])),
                }
            ),
            "containment": containment,
            "descendants": [],
            "state": "launching",
            "created_at_ns": time.time_ns(),
        }
        if boundary is not None:
            record["os_boundary"] = boundary
        try:
            self._write_process_record(record_path, record)
        except BaseException:
            os.close(gate_read)
            os.close(gate_write)
            os.close(ready_read)
            os.close(ready_write)
            os.close(containment_descriptor)
            raise
        process: subprocess.Popen[bytes] | None = None
        selector: selectors.BaseSelector | None = None
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        truncated = {"stdout": False, "stderr": False}
        deadline = time.monotonic() + timeout_seconds
        if self.overall_deadline is not None:
            deadline = min(deadline, self.overall_deadline)
        failure: HarnessError | None = None
        outcome = "failed"
        process_waited = False
        process_event_started = False
        try:
            process = subprocess.Popen(
                launch_argv,
                cwd=str(self.workspace),
                env=environment,
                stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                pass_fds=(gate_read, ready_write, containment_descriptor),
            )
            os.close(gate_read)
            gate_read = -1
            os.close(ready_write)
            ready_write = -1
            os.close(containment_descriptor)
            containment_descriptor = -1
            record.update(
                {
                    "state": "spawned_unverified",
                    "pid": process.pid,
                    "pgid": process.pid,
                    "gate_released": False,
                }
            )
            self._write_process_record(record_path, record)
            containment_deadline = min(deadline, time.monotonic() + 2)
            expected_birth: tuple[str, int] | None = None
            identity: dict[str, Any] | None = None
            early_result: ProcessResult | None = None
            gate_ready = False
            while True:
                current = self._process_identity(process.pid)
                if current is None:
                    brief = self._process_birth_identity(process.pid)
                    if brief is not None:
                        if brief[1] != os.getuid():
                            raise HarnessError("spawned process changed owner before ownership")
                        if expected_birth is None:
                            expected_birth = brief
                        elif brief != expected_birth:
                            raise HarnessError(
                                "spawned process birth identity changed before ownership"
                            )
                else:
                    brief = (current.get("birth"), current.get("uid"))
                    if (
                        current.get("pid") != process.pid
                        or not isinstance(brief[0], str)
                        or brief[1] != os.getuid()
                    ):
                        raise HarnessError("spawned process has an invalid birth identity")
                    if expected_birth is None:
                        expected_birth = brief
                    elif brief != expected_birth:
                        raise HarnessError(
                            "spawned process birth identity changed before ownership"
                        )
                    if "birth_identity" not in record:
                        record["birth_identity"] = current
                        executable = current.get("executable")
                        if isinstance(executable, str):
                            record["allowed_executables"] = sorted(
                                {*record["allowed_executables"], executable}
                            )
                        self._write_process_record(record_path, record)
                if not gate_ready and ready_read >= 0:
                    try:
                        ready = os.read(ready_read, 1)
                    except BlockingIOError:
                        ready = None
                    if ready == b"1":
                        gate_ready = True
                        os.close(ready_read)
                        ready_read = -1
                    elif ready == b"":
                        os.close(ready_read)
                        ready_read = -1
                if current is not None and gate_ready:
                    if (
                        current.get("pgid") == process.pid
                        and self._process_has_marker(process.pid, containment_marker)
                        and self._live_containment_matches(process.pid, record)
                    ):
                        identity = current
                        break
                early_result = self._waited_fast_process_result(process, argv)
                if early_result is not None:
                    process_waited = True
                    break
                if time.monotonic() >= containment_deadline:
                    raise HarnessError("spawned process has no enforceable containment boundary")
                time.sleep(0.005)
            if early_result is not None:
                if early_result.returncode == -signal.SIGXFSZ or b"File too large" in early_result.stderr:
                    raise HarnessError(
                        f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
                    )
                if check and early_result.returncode != 0:
                    message = early_result.stderr.decode("utf-8", errors="replace")[-4000:]
                    raise HarnessError(
                        f"command failed ({early_result.returncode}): {' '.join(argv)}\n{message}"
                    )
                outcome = "completed"
                return early_result
            assert identity is not None
            record.update(
                {
                    "state": "owned",
                    "birth_identity": identity,
                    "current_executable": identity.get("executable"),
                    "observed_cwd": identity.get("cwd"),
                }
            )
            if identity["executable"] not in record["allowed_executables"]:
                record["allowed_executables"] = sorted(
                    {*record["allowed_executables"], identity["executable"]}
            )
            self._write_process_record(record_path, record)
            process_event_started = True
            self._emit_process_event("command_started", record, process.pid)
            record["gate_released"] = True
            self._write_process_record(record_path, record)
            os.write(gate_write, b"1")
            os.close(gate_write)
            gate_write = -1
            assert process.stdout is not None and process.stderr is not None
            if input_bytes is not None:
                assert process.stdin is not None
                try:
                    process.stdin.write(input_bytes)
                    process.stdin.close()
                except BrokenPipeError:
                    pass
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    failure = HarnessError(
                        f"command timed out after {timeout_seconds}s: {' '.join(argv)}"
                    )
                    break
                if selector.get_map():
                    events = selector.select(timeout=min(0.05, remaining))
                else:
                    time.sleep(min(0.05, remaining))
                    events = []
                for key, _mask in events:
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    name = key.data
                    room = MAX_COMMAND_BYTES - len(buffers[name])
                    if room > 0:
                        buffers[name].extend(chunk[:room])
                    if len(chunk) > room:
                        truncated[name] = True
                self._refresh_root_identity(record, record_path)
                exited = process.poll() is not None
                self._track_descendants(
                    process.pid,
                    record,
                    record_path,
                    discover_detached=exited,
                )
                if self._storage_budget_exceeded():
                    failure = HarnessError(
                        f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
                    )
                    break
                if exited:
                    break
            if failure is not None:
                raise failure
            if self._storage_budget_exceeded():
                raise HarnessError(
                    f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
                )
            result = ProcessResult(
                list(argv),
                process.returncode,
                bytes(buffers["stdout"]),
                bytes(buffers["stderr"]),
                truncated["stdout"],
                truncated["stderr"],
            )
            if result.returncode == -signal.SIGXFSZ or b"File too large" in result.stderr:
                raise HarnessError(
                    f"workspace/session storage budget of {self.max_session_bytes} bytes was exceeded"
                )
            if check and result.returncode != 0:
                message = result.stderr.decode("utf-8", errors="replace")[-4000:]
                raise HarnessError(
                    f"command failed ({result.returncode}): {' '.join(argv)}\n{message}"
                )
            outcome = "completed"
            return result
        finally:
            for descriptor in (
                gate_read,
                gate_write,
                ready_read,
                ready_write,
                containment_descriptor,
            ):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if selector is not None:
                selector.close()
            cleanup_error = None
            if process is not None:
                if not process_waited:
                    cleanup_error = self._kill_and_reap(process, record, record_path)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                record.update(
                    {
                        "state": outcome if cleanup_error is None else "cleanup_failed",
                        "returncode": process.returncode,
                        "finished_at_ns": time.time_ns(),
                    }
                )
            else:
                record.update({"state": "spawn_failed", "finished_at_ns": time.time_ns()})
            self._write_process_record(record_path, record)
            if process is not None and process_event_started:
                self._emit_process_event("command_finished", record, None)
            if cleanup_error is not None and sys.exc_info()[0] is None:
                raise cleanup_error

    def _storage_budget_exceeded(self) -> bool:
        _validate_session_tree(self.session_root)
        session_bytes, session_entries = self._tree_usage(
            self.session_root,
            self.max_session_bytes + 1,
            self.max_storage_entries + 1,
        )
        if session_bytes > self.max_session_bytes or session_entries > self.max_storage_entries:
            return True
        workspace_bytes, workspace_entries = self._tree_usage(
            self.workspace,
            self.workspace_baseline_bytes + self.max_session_bytes + 1,
            self.workspace_baseline_entries + self.max_storage_entries + 1,
            excluded=(self.workspace / ".git",),
        )
        return (
            workspace_bytes - self.workspace_baseline_bytes > self.max_session_bytes
            or workspace_entries - self.workspace_baseline_entries > self.max_storage_entries
        )

    @staticmethod
    def _same_process(
        expected: dict[str, Any],
        current: dict[str, Any] | None,
    ) -> bool:
        return (
            current is not None
            and current.get("pid") == expected.get("pid")
            and current.get("birth") == expected.get("birth")
            and current.get("uid") == expected.get("uid") == os.getuid()
        )

    def _owned_descendant_matches(
        self,
        record: dict[str, Any],
        expected: dict[str, Any],
        current: dict[str, Any] | None,
    ) -> bool:
        if not self._same_process(expected, current):
            return False
        if record.get("containment") == "darwin_sandbox":
            try:
                return self._darwin_boundary_matches(current["pid"], record)
            except HarnessError:
                return False
        return True

    def _quiesce_owned_process_group(
        self,
        process: subprocess.Popen[bytes],
        record: dict[str, Any],
    ) -> HarnessError | None:
        if process.returncode is not None or record.get("state") != "owned":
            return None
        expected = record.get("birth_identity")
        if (
            record.get("pid") != process.pid
            or record.get("pgid") != process.pid
            or record.get("workspace") != str(self.workspace)
            or record.get("session_root") != str(self.session_root)
            or record.get("session_id") != self.session_root.name
            or not isinstance(expected, dict)
            or expected.get("pid") != process.pid
            or expected.get("uid") != os.getuid()
            or not self._record_session_marker_matches(record)
        ):
            return HarnessError("refused to quiesce a process without exact owned identity")
        try:
            current = self._process_identity(process.pid)
            brief = self._process_birth_identity(process.pid) if current is None else None
        except HarnessError as error:
            return error
        if current is not None and not self._identity_matches(record, current):
            return HarnessError(
                f"refused to quiesce pid {process.pid} after birth identity changed"
            )
        if brief is not None and brief != (expected.get("birth"), expected.get("uid")):
            return HarnessError(
                f"refused to quiesce pid {process.pid} after birth identity changed"
            )
        try:
            os.killpg(process.pid, signal.SIGSTOP)
        except ProcessLookupError:
            return None
        except PermissionError as error:
            return HarnessError(f"cannot stop owned process group {process.pid}: {error}")
        stopped = False
        wait_error: BaseException | None = None
        kill_error: BaseException | None = None
        continue_error: BaseException | None = None
        reap_error: BaseException | None = None
        try:
            deadline = time.monotonic() + 0.2
            while time.monotonic() < deadline:
                status = os.waitid(
                    os.P_PID,
                    process.pid,
                    os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT,
                )
                if status is not None:
                    stopped = True
                    break
                time.sleep(0.001)
        except ChildProcessError:
            stopped = True
        except BaseException as error:
            wait_error = error
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except BaseException as error:
                kill_error = error
            if kill_error is None:
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired as error:
                    reap_error = error
                except BaseException as error:
                    reap_error = error
            else:
                try:
                    os.killpg(process.pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    continue_error = error
        if kill_error is not None:
            message = f"cannot kill quiesced process group {process.pid}: {kill_error}"
            if continue_error is not None:
                message += f"; cannot resume it after failed kill: {continue_error}"
            return HarnessError(message)
        if wait_error is not None:
            if isinstance(wait_error, (KeyboardInterrupt, SystemExit)):
                raise wait_error
            return HarnessError(
                f"cannot confirm stopped process group {process.pid}: "
                f"{type(wait_error).__name__}: {wait_error}"
            )
        if reap_error is not None:
            return HarnessError(
                f"cannot reap quiesced process group {process.pid}: "
                f"{type(reap_error).__name__}: {reap_error}"
            )
        if not stopped:
            return HarnessError(f"owned process group {process.pid} did not quiesce before cleanup")
        return None

    def _kill_and_reap(
        self,
        process: subprocess.Popen[bytes],
        record: dict[str, Any],
        record_path: Path,
    ) -> HarnessError | None:
        errors: list[str] = []

        if (
            record.get("state") in {"launching", "spawned_unverified"}
            and record.get("pid") == process.pid
            and record.get("pgid") == process.pid
            and record.get("gate_released") is False
        ):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    errors.append(f"cannot kill gated process group {process.pid}: {error}")
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                errors.append(f"failed to reap gated process group leader {process.pid}")
            if errors:
                return HarnessError("; ".join(errors))
            return None

        quiesce_error = self._quiesce_owned_process_group(process, record)
        if quiesce_error is not None:
            errors.append(str(quiesce_error))

        def inspect(pid: int) -> tuple[dict[str, Any] | None, bool]:
            try:
                return self._process_identity(pid), False
            except HarnessError as error:
                message = str(error)
                if message not in errors:
                    errors.append(message)
                return None, True

        def kill_descendants(descendants: list[dict[str, Any]]) -> list[dict[str, Any]]:
            live = []
            for expected in reversed(descendants):
                pid = expected.get("pid")
                if not isinstance(pid, int):
                    continue
                current, _inspection_failed = inspect(pid)
                if not self._owned_descendant_matches(record, expected, current):
                    continue
                live.append(expected)
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    message = f"cannot kill descendant {pid}: {error}"
                    if message not in errors:
                        errors.append(message)
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass
            return live

        # Recorded detached descendants must stop running before any cleanup
        # discovery can enter a fallible path.  Cleanup updates the record only
        # in memory; persistence and event callbacks happen after containment.
        known_descendants = [
            item for item in record.get("descendants", []) if isinstance(item, dict)
        ]
        kill_descendants(known_descendants)

        deadline = time.monotonic() + 2
        quiet_passes = 0
        remaining: list[dict[str, Any]] = []
        while time.monotonic() < deadline:
            try:
                descendants = self._discover_descendants(
                    process.pid,
                    record,
                    discover_detached=True,
                )
            except HarnessError as error:
                if str(error) not in errors:
                    errors.append(str(error))
                descendants = [
                    item for item in record.get("descendants", []) if isinstance(item, dict)
                ]
            else:
                record["descendants"] = descendants
            remaining = kill_descendants(descendants)
            process.poll()
            current_root, root_inspection_failed = inspect(process.pid)
            if current_root is not None and self._identity_matches(record, current_root):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    message = f"cannot kill process group {process.pid}: {error}"
                    if message not in errors:
                        errors.append(message)
            elif root_inspection_failed and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as error:
                    message = f"cannot kill owned process group {process.pid}: {error}"
                    if message not in errors:
                        errors.append(message)
            elif current_root is not None and process.returncode is None:
                message = f"refused to signal pid {process.pid} after birth identity changed"
                if message not in errors:
                    errors.append(message)
            if process.returncode is not None and not remaining:
                quiet_passes += 1
                if quiet_passes >= 3:
                    break
            else:
                quiet_passes = 0
            time.sleep(0.01)
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            errors.append(f"failed to reap process group leader {process.pid}")
        remaining = []
        for expected in record.get("descendants", []):
            if not isinstance(expected, dict) or not isinstance(expected.get("pid"), int):
                continue
            current, inspection_failed = inspect(expected["pid"])
            if inspection_failed or self._owned_descendant_matches(record, expected, current):
                remaining.append(expected)
        if remaining:
            errors.append(
                "failed to contain descendants: "
                + ",".join(str(item.get("pid")) for item in remaining)
            )
        if errors:
            return HarnessError("; ".join(errors))
        return None

    def _write_process_record(self, path: Path, record: dict[str, Any]) -> None:
        encoded = (canonical_json(record) + "\n").encode()
        current = self._tree_bytes(self.session_root, self.max_session_bytes + 1)
        if current + len(encoded) > self.max_session_bytes:
            raise HarnessError("session storage budget would be exceeded by process ownership")
        try:
            relative = path.relative_to(self.session_root)
        except ValueError as error:
            raise HarnessError("process ownership record escapes session storage") from error
        _session_atomic_bytes(self.session_root, relative, encoded)

    def _emit_process_event(
        self,
        event_type: str,
        record: dict[str, Any],
        pid: int | None,
    ) -> None:
        if self.process_event_callback is None:
            return
        self.process_event_callback(
            {
                "type": event_type,
                "process_id": record["process_id"],
                "purpose": record["purpose"],
                "owner": record["owner"],
                "pid": record.get("pid"),
                "pgid": record.get("pgid"),
                "state": record["state"],
            },
            pid,
        )

    def _recover_owned_processes(self) -> None:
        for path in sorted(self.process_root.glob("*.json")):
            if path.is_symlink():
                raise HarnessError("process ownership record is a symlink")
            try:
                record = json.loads(
                    _session_read_bytes(self.session_root, path.relative_to(self.session_root))
                )
            except (OSError, json.JSONDecodeError) as error:
                raise HarnessError(f"invalid process ownership record: {path.name}") from error
            if record.get("state") not in {
                "launching",
                "spawned_unverified",
                "owned",
                "cleanup_failed",
            }:
                continue
            pid = record.get("pid")
            pgid = record.get("pgid")
            if not isinstance(pid, int):
                record.update({"state": "abandoned_before_spawn", "finished_at_ns": time.time_ns()})
                self._write_process_record(path, record)
                continue
            identity = self._process_identity(pid)
            if identity is None:
                record.update({"state": "exited_before_resume", "finished_at_ns": time.time_ns()})
                self._write_process_record(path, record)
                continue
            if not isinstance(pgid, int) or pgid != pid or not self._identity_matches(record, identity):
                record.update({"state": "stale_identity_mismatch", "finished_at_ns": time.time_ns()})
                self._write_process_record(path, record)
                raise HarnessError(
                    f"refused to signal stale process record with mismatched birth identity: {pid}"
                )
            descendants = {
                item["pid"]: item
                for item in record.get("descendants", [])
                if isinstance(item, dict) and isinstance(item.get("pid"), int)
            }
            deadline = time.monotonic() + 2
            quiet_passes = 0
            while time.monotonic() < deadline:
                self._refresh_root_identity(record, path)
                for marked in self._marked_processes(record):
                    if marked["pid"] != pid:
                        descendants[marked["pid"]] = marked
                record["descendants"] = [descendants[key] for key in sorted(descendants)]
                self._write_process_record(path, record)
                active = []
                for descendant in reversed(record["descendants"]):
                    child = descendant["pid"]
                    current = self._process_identity(child)
                    if not self._owned_descendant_matches(record, descendant, current):
                        continue
                    active.append(descendant)
                    try:
                        os.kill(child, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError as error:
                        raise HarnessError(
                            f"cannot terminate recovered descendant {child}"
                        ) from error
                    try:
                        os.waitpid(child, os.WNOHANG)
                    except ChildProcessError:
                        pass
                current = self._process_identity(pid)
                root_active = current is not None and self._identity_matches(record, current)
                if root_active:
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        root_active = False
                    except PermissionError as error:
                        raise HarnessError(
                            f"cannot terminate recovered process group {pgid}"
                        ) from error
                if not root_active and not active:
                    quiet_passes += 1
                    if quiet_passes >= 3:
                        break
                else:
                    quiet_passes = 0
                time.sleep(0.01)
            else:
                raise HarnessError(f"recovered process group {pgid} or descendants did not terminate")
            record.update({"state": "killed_on_resume", "finished_at_ns": time.time_ns()})
            self._write_process_record(path, record)

    @staticmethod
    def _tree_bytes(root: Path, stop_after: int, *, excluded: Sequence[Path] = ()) -> int:
        return WorkspaceTools._tree_usage(
            root,
            stop_after,
            (1 << 63) - 1,
            excluded=excluded,
        )[0]

    @staticmethod
    def _tree_usage(
        root: Path,
        stop_after_bytes: int,
        stop_after_entries: int,
        *,
        excluded: Sequence[Path] = (),
    ) -> tuple[int, int]:
        total = 0
        count = 0
        pending = [root]
        excluded_paths = set(excluded)
        while pending:
            current = pending.pop()
            if current in excluded_paths:
                continue
            try:
                entries = os.scandir(current)
            except FileNotFoundError:
                continue
            except PermissionError as error:
                raise HarnessError(f"storage accounting cannot read directory: {current}") from error
            with entries:
                for entry in entries:
                    try:
                        count += 1
                        if entry.is_symlink():
                            total += len(os.fsencode(os.readlink(entry.path)))
                        elif entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                        if total >= stop_after_bytes or count >= stop_after_entries:
                            return total, count
                    except FileNotFoundError:
                        continue
                    except PermissionError as error:
                        raise HarnessError(
                            f"storage accounting cannot inspect path: {entry.path}"
                        ) from error
        return total, count


class CodexHarnessRunner:
    """Durable provider-neutral tool loop supervised by the Codex coordinator."""

    provider_id = "oxalpha-provider-pool"
    requires_receipted_tests = True

    def __init__(
        self,
        pool: Any,
        session_dir: Path,
        timeout_seconds: int,
        *,
        race_failure_type: type[BaseException],
        max_turns: int = 128,
        max_output_tokens: int = 32768,
        max_context_bytes: int = MAX_SESSION_CONTEXT_BYTES,
        max_session_bytes: int = MAX_SESSION_ARTIFACT_BYTES,
        max_storage_entries: int = MAX_STORAGE_ENTRIES,
    ):
        self.pool = pool
        _reject_symlink_components(session_dir)
        absolute_session_dir = session_dir.absolute()
        self.session_dir = absolute_session_dir.parent.resolve() / absolute_session_dir.name
        self.session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _validate_session_tree(self.session_dir)
        os.chmod(self.session_dir, 0o700)
        self.timeout_seconds = timeout_seconds
        self.race_failure_type = race_failure_type
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_context_bytes = max_context_bytes
        self.max_session_bytes = max_session_bytes
        self.max_storage_entries = max_storage_entries
        self.model = f"oxalpha-race/{pool.settings.virtual_model}"

    def verify_test_receipts(
        self,
        result: HarnessRunResult,
        workspace: Path,
        task: dict[str, Any],
        role: str,
    ) -> tuple[dict[str, Any], ...]:
        session_id = result.session_id
        if not isinstance(session_id, str) or not SESSION_ID_RE.fullmatch(session_id):
            raise HarnessError("test receipts have no valid native session")
        root = self.session_dir / session_id
        _validate_session_tree(root)
        try:
            state = json.loads(_session_read_bytes(root, Path("state.json")))
        except (OSError, json.JSONDecodeError) as error:
            raise HarnessError("test receipt session state is malformed") from error
        if state.get("workspace") != str(workspace.resolve()) or state.get("role") != role:
            raise HarnessError("test receipt session identity mismatch")
        tools = WorkspaceTools(
            workspace,
            role,
            task,
            root,
            max_session_bytes=self.max_session_bytes,
            max_storage_entries=self.max_storage_entries,
        )
        fingerprint = tools.workspace_fingerprint()
        generation = state.get("mutation_generation", 0)
        state_receipts = {
            item.get("command"): item
            for item in state.get("test_receipts", [])
            if isinstance(item, dict) and isinstance(item.get("command"), str)
        }
        verified = []
        for receipt in result.test_receipts:
            if not isinstance(receipt, dict) or receipt != state_receipts.get(receipt.get("command")):
                raise HarnessError("test receipt does not match durable session state")
            artifact = receipt.get("artifact")
            if not isinstance(artifact, str) or not re.fullmatch(
                r"tool-results/[0-9A-Za-z_.-]+\.json", artifact
            ):
                raise HarnessError("test receipt artifact name is invalid")
            try:
                encoded = _session_read_bytes(root, Path(artifact))
            except OSError as error:
                raise HarnessError("test receipt artifact is missing or unsafe") from error
            payload = json.loads(encoded)
            digest = sha256_bytes(encoded)
            if digest != receipt.get("sha256"):
                raise HarnessError("test receipt artifact hash mismatch")
            if (
                payload.get("declared_test") is not True
                or payload.get("command") != receipt.get("command")
                or payload.get("exit_code") != receipt.get("exit_code")
                or receipt.get("mutation_generation") != generation
                or receipt.get("workspace_fingerprint") != fingerprint
            ):
                raise HarnessError("test receipt contents or workspace fingerprint mismatch")
            current = dict(receipt)
            current["controller_verified"] = True
            verified.append(current)
        return tuple(verified)

    def run(
        self,
        workspace: Path,
        prompt: str,
        *,
        role: str,
        task: dict[str, Any],
        session_id: str | None,
        event_callback: EventCallback,
    ) -> HarnessRunResult:
        started = time.monotonic()
        event_lines: list[str] = []
        event_output_bytes = 0
        event_output_truncated = False
        state, root = self._load_or_create(workspace, role, task, session_id)
        session_id = state["session_id"]

        def emit(event: dict[str, Any], pid: int | None = None) -> None:
            nonlocal event_output_bytes, event_output_truncated
            encoded = canonical_json(event) + "\n"
            raw = encoded.encode("utf-8")
            if event_output_bytes + len(raw) <= MAX_EVENT_OUTPUT_BYTES:
                event_lines.append(encoded)
                event_output_bytes += len(raw)
            elif not event_output_truncated:
                marker = canonical_json({"type": "event_output_truncated"}) + "\n"
                if event_output_bytes + len(marker.encode()) <= MAX_EVENT_OUTPUT_BYTES:
                    event_lines.append(marker)
                event_output_truncated = True
            self._reserve_session_bytes(root, len(raw))
            _session_append_record(root, Path("journal.jsonl"), raw)
            event_callback(encoded, event, pid)

        emit({"type": "session_resumed" if state["turn"] else "session_started", "session_id": session_id})
        tools = WorkspaceTools(
            workspace,
            role,
            task,
            root,
            max_session_bytes=self.max_session_bytes,
            max_storage_entries=self.max_storage_entries,
            overall_deadline=started + self.timeout_seconds,
            process_event_callback=lambda event, pid: emit(
                {"session_id": session_id, **event}, pid
            ),
        )
        self._execute_pending_tools(state, root, tools, emit)
        completed = self._final_result(state, tools, event_lines)
        if completed is not None:
            return completed
        if state["awaiting_model_response"]:
            emit(
                {
                    "type": "pending_model_turn_resumed",
                    "session_id": session_id,
                    "turn": int(state["turn"]) + 1,
                }
            )
        else:
            state["messages"].append({"role": "user", "content": prompt})
            state["awaiting_model_response"] = True
            self._save_state(root, state)
        while state["turn"] < self.max_turns:
            if time.monotonic() - started > self.timeout_seconds:
                emit({"type": "agent_timeout", "session_id": session_id})
                return HarnessRunResult(124, "", _bounded_event_output(event_lines), session_id, state["tokens"], timed_out=True)
            turn = int(state["turn"]) + 1
            self._compact_context(state, root)
            context_key = hashlib.sha256(
                f"{workspace.resolve()}\0{role}\0{session_id}".encode("utf-8")
            ).hexdigest()[:32]
            body = {
                "model": self.pool.settings.virtual_model,
                "messages": state["messages"],
                "tools": tools.schemas(),
                "tool_choice": "auto",
                "temperature": 0.1,
                "max_tokens": self.max_output_tokens,
                "stream": False,
            }
            emit({"type": "model_turn_started", "session_id": session_id, "turn": turn})
            try:
                request_id, result = self.pool.race(body, context_key)
            except self.race_failure_type as error:
                emit(
                    {
                        "type": "provider_error",
                        "session_id": session_id,
                        "turn": turn,
                        "error": str(error)[:2000],
                    }
                )
                return HarnessRunResult(
                    75,
                    "",
                    _bounded_event_output(event_lines, f"temporary provider unavailable: {error}\n"),
                    session_id,
                    state["tokens"],
                )
            response_artifact = self._archive_response(
                state, root, turn, request_id, result.provider_id, result.body
            )
            try:
                response = json.loads(result.body)
                choice = response["choices"][0]
                message = choice["message"]
            except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
                emit({"type": "malformed_provider_response", "session_id": session_id, "turn": turn})
                return HarnessRunResult(
                    75,
                    "",
                    _bounded_event_output(event_lines, f"malformed provider response: {error}\n"),
                    session_id,
                    state["tokens"],
                    malformed_json_lines=1,
                )
            try:
                assistant, calls = self._assistant_message(message)
            except HarnessError as error:
                emit(
                    {
                        "type": "malformed_provider_response",
                        "session_id": session_id,
                        "turn": turn,
                        "error": str(error)[:2000],
                    }
                )
                return HarnessRunResult(
                    75,
                    "",
                    _bounded_event_output(event_lines, f"malformed provider response: {error}\n"),
                    session_id,
                    state["tokens"],
                    malformed_json_lines=1,
                )
            state["messages"].append(assistant)
            state["turn"] = turn
            state["tokens"] += self._usage_tokens(response.get("usage"))
            state["awaiting_model_response"] = False
            state["pending_tool_calls"] = [
                {"turn": turn, "index": index, "call": call, "status": "pending"}
                for index, call in enumerate(calls, start=1)
            ]
            self._save_state(root, state)
            emit(
                {
                    "type": "model_turn_finished",
                    "session_id": session_id,
                    "turn": turn,
                    "provider": result.provider_id,
                    "request_id": request_id,
                    "finish_reason": choice.get("finish_reason"),
                    "tool_calls": len(calls),
                    "response_artifact": response_artifact,
                    "tokens": state["tokens"],
                }
            )
            if not calls:
                content = assistant.get("content")
                text = content if isinstance(content, str) else ""
                return HarnessRunResult(
                    0,
                    text.strip(),
                    _bounded_event_output(event_lines),
                    session_id,
                    state["tokens"],
                    test_receipts=self._current_test_receipts(
                        state,
                        tools.workspace_fingerprint(),
                    ),
                )
            self._execute_pending_tools(state, root, tools, emit)
            completed = self._final_result(state, tools, event_lines)
            if completed is not None:
                return completed
        emit({"type": "agent_turn_limit", "session_id": session_id, "max_turns": self.max_turns})
        return HarnessRunResult(
            2,
            "agent turn limit exceeded",
            _bounded_event_output(event_lines),
            session_id,
            state["tokens"],
        )

    def _load_or_create(
        self,
        workspace: Path,
        role: str,
        task: dict[str, Any],
        session_id: str | None,
    ) -> tuple[dict[str, Any], Path]:
        workspace_value = str(workspace.resolve())
        if session_id is None:
            session_id = "codex-oxalpha-" + secrets.token_hex(16)
            root = self.session_dir / session_id
            root.mkdir(mode=0o700)
            _validate_session_tree(root)
            for relative in (
                Path("responses"),
                Path("tool-results"),
                Path("tool-receipts"),
                Path("context-archives"),
            ):
                _session_mkdir(root, relative)
            state = {
                "schema_version": 1,
                "session_id": session_id,
                "workspace": workspace_value,
                "role": role,
                "task_id": task.get("id"),
                "turn": 0,
                "tokens": 0,
                "mutation_generation": 0,
                "test_receipts": [],
                "awaiting_model_response": False,
                "context_archive_count": 0,
                "pending_tool_calls": [],
                "messages": [{"role": "system", "content": self._system_prompt(role)}],
            }
            self._save_state(root, state)
            return state, root
        if not SESSION_ID_RE.fullmatch(session_id):
            raise HarnessError("invalid native harness session id")
        root = self.session_dir / session_id
        _validate_session_tree(root)
        self._recover_journal(root)
        try:
            state = json.loads(_session_read_bytes(root, Path("state.json")))
        except (OSError, json.JSONDecodeError) as error:
            raise HarnessError("session state is malformed") from error
        if state.get("workspace") != workspace_value or state.get("role") != role:
            raise HarnessError("session workspace or role mismatch")
        if state.get("task_id") != task.get("id"):
            raise HarnessError("session task mismatch")
        if not isinstance(state.get("messages"), list):
            raise HarnessError("session messages are malformed")
        if not isinstance(state.get("pending_tool_calls", []), list):
            raise HarnessError("session pending tool calls are malformed")
        for relative in (
            Path("responses"),
            Path("tool-results"),
            Path("tool-receipts"),
            Path("context-archives"),
        ):
            _session_mkdir(root, relative)
        _validate_session_tree(root)
        state.setdefault("pending_tool_calls", [])
        state.setdefault("mutation_generation", 0)
        state.setdefault("test_receipts", [])
        state.setdefault("context_archive_count", 0)
        if "awaiting_model_response" not in state:
            state["awaiting_model_response"] = bool(
                state["messages"] and state["messages"][-1].get("role") in {"user", "tool"}
            )
        return state, root

    def _recover_journal(self, root: Path) -> None:
        relative = Path("journal.jsonl")
        try:
            encoded = _session_read_bytes(root, relative)
        except FileNotFoundError:
            return
        if len(encoded) > self.max_session_bytes:
            raise HarnessError("journal exceeds the session artifact budget")
        complete = encoded
        if encoded and not encoded.endswith(b"\n"):
            boundary = encoded.rfind(b"\n") + 1
            complete = encoded[:boundary]
            _session_truncate(root, relative, boundary)
        for line_number, line in enumerate(complete.splitlines(), start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise HarnessError(
                    f"journal contains a malformed complete record at line {line_number}"
                ) from error
            if not isinstance(event, dict):
                raise HarnessError(
                    f"journal contains a non-object record at line {line_number}"
                )

    def _execute_pending_tools(
        self,
        state: dict[str, Any],
        root: Path,
        tools: WorkspaceTools,
        emit: Callable[[dict[str, Any]], None],
    ) -> None:
        while state["pending_tool_calls"]:
            pending = state["pending_tool_calls"][0]
            call = pending.get("call")
            if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                raise HarnessError("pending tool call is malformed")
            turn = pending.get("turn")
            index = pending.get("index")
            if not isinstance(turn, int) or not isinstance(index, int):
                raise HarnessError("pending tool call position is malformed")
            tool_name = call["function"].get("name")
            if not isinstance(tool_name, str) or not tool_name:
                raise HarnessError("pending tool call has no name")
            artifact = self._tool_artifact_path(turn, index, tool_name)
            artifact_path = root / artifact
            payload = None
            recovered = False
            mutating = tool_name in {"apply_patch", "run_command"}
            arguments_text = call["function"].get("arguments", "{}")
            if not isinstance(arguments_text, str):
                arguments_text = canonical_json(arguments_text)
            arguments_sha256 = sha256_bytes(arguments_text.encode())
            tools.tool_context = {
                "turn": turn,
                "index": index,
                "tool": tool_name,
                "call_id": call.get("id"),
            }
            if pending.get("status") == "executing":
                payload = self._load_tool_completion(
                    root,
                    turn,
                    index,
                    tool_name,
                    call,
                    arguments_sha256,
                    tools,
                )
                recovered = payload is not None
            if pending.get("status") == "executing" and payload is None and mutating:
                payload = {
                    "ok": False,
                    "error": "prior tool execution was interrupted; side effects are uncertain",
                    "recovery_required": True,
                }
                self._archive_tool_result(root, turn, index, tool_name, payload)
                self._archive_tool_completion(
                    root,
                    turn,
                    index,
                    tool_name,
                    call,
                    arguments_sha256,
                    payload,
                    tools.workspace_fingerprint(),
                )
                recovered = True
            if payload is None:
                pending["status"] = "executing"
                pending["arguments_sha256"] = arguments_sha256
                self._save_state(root, state)
                try:
                    arguments = json.loads(arguments_text)
                    if not isinstance(arguments, dict):
                        raise HarnessError("tool arguments must decode to an object")
                    payload = tools.execute(tool_name, arguments)
                except (HarnessError, json.JSONDecodeError, OSError, ValueError) as error:
                    payload = {"ok": False, "error": f"{type(error).__name__}: {error}"}
                self._archive_tool_result(root, turn, index, tool_name, payload)
                self._archive_tool_completion(
                    root,
                    turn,
                    index,
                    tool_name,
                    call,
                    arguments_sha256,
                    payload,
                    tools.workspace_fingerprint(),
                )
            if tool_name == "apply_patch" and (
                payload.get("ok") is True or payload.get("recovery_required") is True
            ):
                state["mutation_generation"] = int(state["mutation_generation"]) + 1
                state["test_receipts"] = []
            if tool_name == "run_command" and payload.get("recovery_required") is True:
                state["mutation_generation"] = int(state["mutation_generation"]) + 1
                state["test_receipts"] = []
            if tool_name == "run_command" and payload.get("declared_test") is True:
                receipt = {
                    "command": payload.get("command"),
                    "exit_code": payload.get("exit_code"),
                    "artifact": artifact,
                    "sha256": sha256_bytes(_session_read_bytes(root, Path(artifact))),
                    "mutation_generation": state["mutation_generation"],
                    "workspace_fingerprint": tools.workspace_fingerprint(),
                }
                state["test_receipts"] = [
                    item
                    for item in state["test_receipts"]
                    if item.get("command") != receipt["command"]
                ] + [receipt]
            if tool_name == "finish_task" and payload.get("ok") is True:
                contract = payload.get("contract")
                if not isinstance(contract, dict):
                    raise HarnessError("finish_task did not return a contract")
                state["final_contract"] = contract
            tools.tool_context = None
            context_payload = self._context_tool_payload(payload, artifact)
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": tool_name,
                    "content": canonical_json(context_payload),
                }
            )
            state["awaiting_model_response"] = True
            state["pending_tool_calls"].pop(0)
            self._save_state(root, state)
            emit(
                {
                    "type": "tool_finished",
                    "session_id": state["session_id"],
                    "turn": turn,
                    "tool": tool_name,
                    "ok": payload.get("ok") is True,
                    "artifact": artifact,
                    "recovered": recovered,
                }
            )

    @staticmethod
    def _final_result(
        state: dict[str, Any],
        tools: WorkspaceTools,
        event_lines: list[str],
    ) -> HarnessRunResult | None:
        contract = state.get("final_contract")
        if not isinstance(contract, dict):
            return None
        return HarnessRunResult(
            0,
            canonical_json(contract),
            _bounded_event_output(event_lines),
            state["session_id"],
            state["tokens"],
            test_receipts=CodexHarnessRunner._current_test_receipts(
                state,
                tools.workspace_fingerprint(),
            ),
        )

    def _compact_context(self, state: dict[str, Any], root: Path) -> None:
        messages = state["messages"]
        if len(canonical_json(messages).encode("utf-8")) <= self.max_context_bytes:
            return
        if not messages or messages[0].get("role") != "system":
            raise HarnessError("session context has no leading system message")
        first_user = next(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            None,
        )
        if first_user is None:
            raise HarnessError("session context has no user task")
        fixed = messages[: first_user + 1]
        groups = []
        index = first_user + 1
        while index < len(messages):
            end = index + 1
            if messages[index].get("role") == "assistant":
                while end < len(messages) and messages[end].get("role") == "tool":
                    end += 1
            groups.append(messages[index:end])
            index = end
        if not groups:
            raise HarnessError("initial task context exceeds the aggregate context limit")
        archive_number = int(state.get("context_archive_count", 0)) + 1
        relative = Path("context-archives") / f"archive-{archive_number:04d}.json"
        summary = {
            "role": "system",
            "content": (
                "Earlier conversation messages were compacted into local artifact "
                f"{relative.as_posix()} (messages=0000000000, sha256={'0' * 64}). "
                "Raw provider responses and tool results remain in their "
                "session artifacts. Continue from the retained recent messages and current workspace."
            ),
        }
        kept = []
        for group in reversed(groups):
            candidate = fixed + [summary] + [item for current in reversed(kept + [group]) for item in current]
            if len(canonical_json(candidate).encode("utf-8")) > self.max_context_bytes:
                break
            kept.append(group)
        kept.reverse()
        if not kept:
            raise HarnessError("latest model turn exceeds the aggregate context limit")
        dropped_count = len(groups) - len(kept)
        if dropped_count <= 0:
            raise HarnessError("session context exceeds the limit and cannot be compacted safely")
        dropped = [item for group in groups[:dropped_count] for item in group]
        archive = {
            "schema_version": 1,
            "session_id": state["session_id"],
            "message_count": len(dropped),
            "messages": dropped,
        }
        archive_digest = sha256_bytes(canonical_json(archive).encode("utf-8"))
        summary["content"] = (
            "Earlier conversation messages were compacted into local artifact "
            f"{relative.as_posix()} (messages={len(dropped):010d}, sha256={archive_digest}). "
            "Raw provider responses and tool results remain in their session artifacts. "
            "Continue from the retained recent messages and current workspace."
        )
        compacted = fixed + [summary] + [item for group in kept for item in group]
        if len(canonical_json(compacted).encode("utf-8")) > self.max_context_bytes:
            raise HarnessError("compacted session context still exceeds the aggregate limit")
        self._archive_json(root, relative, archive)
        state["messages"] = compacted
        state["context_archive_count"] = archive_number
        self._save_state(root, state)

    def _session_bytes(self, root: Path) -> int:
        _validate_session_tree(root)
        return WorkspaceTools._tree_usage(
            root,
            (1 << 63) - 1,
            (1 << 63) - 1,
        )[0]

    def _session_entries(self, root: Path) -> int:
        _validate_session_tree(root)
        return WorkspaceTools._tree_usage(
            root,
            (1 << 63) - 1,
            (1 << 63) - 1,
        )[1]

    def _reserve_session_bytes(self, root: Path, additional: int) -> None:
        if (
            additional < 0
            or self._session_bytes(root) + additional > self.max_session_bytes
            or self._session_entries(root) + 1 > self.max_storage_entries
        ):
            raise HarnessError(
                f"session artifact budget of {self.max_session_bytes} bytes would be exceeded"
            )

    def _archive_json(self, root: Path, relative: Path, payload: Any) -> None:
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        self._reserve_session_bytes(root, len(encoded))
        _session_atomic_bytes(root, relative, encoded)

    @staticmethod
    def _current_test_receipts(
        state: dict[str, Any], workspace_fingerprint: str
    ) -> tuple[dict[str, Any], ...]:
        generation = state.get("mutation_generation", 0)
        return tuple(
            dict(item)
            for item in state.get("test_receipts", [])
            if item.get("mutation_generation") == generation
            and item.get("workspace_fingerprint") == workspace_fingerprint
        )

    @staticmethod
    def _system_prompt(role: str) -> str:
        edit_rule = (
            "Use apply_patch for all edits and stay inside the declared write_set."
            if role == "implementer"
            else "You are read-only: never modify files or attempt to repair the candidate."
        )
        return (
            "You are one durable coding-agent thread supervised by Codex. The provider serving any "
            "turn may change, so rely only on the supplied conversation and tool results. Inspect actual "
            "source before making semantic claims. Use the available tools instead of inventing file "
            f"contents. {edit_rule} For apply_patch, send a git-apply-compatible patch beginning with "
            "'diff --git'; never send '*** Begin Patch' syntax. Raw outputs are stored as session "
            "artifacts; use read_artifact when "
            "a compact tool result says more bytes are available. Execute tools sequentially. Finish with "
            "finish_task as the only tool call in the final turn; its arguments are the exact final "
            "contract. Never finish with prose or a fenced JSON block."
        )

    @staticmethod
    def _assistant_message(message: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if not isinstance(message, dict):
            raise HarnessError("assistant message is not an object")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            content = canonical_json(content)
        raw_calls = message.get("tool_calls", [])
        if raw_calls is None:
            raw_calls = []
        if not isinstance(raw_calls, list):
            raise HarnessError("tool_calls is not an array")
        if len(raw_calls) > MAX_TOOL_CALLS_PER_TURN:
            raise HarnessError("too many tool calls in one turn")
        calls = []
        for index, value in enumerate(raw_calls, start=1):
            if not isinstance(value, dict) or not isinstance(value.get("function"), dict):
                raise HarnessError("malformed tool call")
            function = value["function"]
            name = function.get("name")
            arguments = function.get("arguments", "{}")
            if not isinstance(name, str) or not name:
                raise HarnessError("tool call is missing a name")
            if isinstance(arguments, dict):
                arguments = canonical_json(arguments)
            if not isinstance(arguments, str):
                raise HarnessError("tool call arguments are malformed")
            call_id = value.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"call-{index}-{sha256_bytes((name + arguments).encode())[:12]}"
            calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        if any(call["function"]["name"] == "finish_task" for call in calls) and len(calls) != 1:
            raise HarnessError("finish_task must be the only tool call in its turn")
        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        return assistant, calls

    @staticmethod
    def _usage_tokens(usage: Any) -> int:
        if not isinstance(usage, dict):
            return 0
        total = usage.get("total_tokens")
        if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
            return total
        answer = 0
        for key in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens"):
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                answer += value
        return answer

    def _save_state(self, root: Path, state: dict[str, Any]) -> None:
        encoded = (canonical_json(state) + "\n").encode("utf-8")
        self._reserve_session_bytes(root, len(encoded))
        _session_atomic_bytes(root, Path("state.json"), encoded)

    def _archive_response(
        self,
        state: dict[str, Any],
        root: Path,
        turn: int,
        request_id: str,
        provider_id: str,
        body: bytes,
    ) -> str:
        safe_provider = re.sub(r"[^A-Za-z0-9_.-]", "_", provider_id)[:80]
        safe_request = re.sub(r"[^A-Za-z0-9_.-]", "_", request_id)[:80]
        relative = Path("responses") / f"turn-{turn:04d}-{safe_request}-{safe_provider}.json"
        self._reserve_session_bytes(root, len(body))
        _session_exclusive_bytes(root, relative, body)
        return relative.as_posix()

    def _archive_tool_result(
        self,
        root: Path,
        turn: int,
        index: int,
        tool_name: str,
        payload: dict[str, Any],
    ) -> str:
        relative = Path(CodexHarnessRunner._tool_artifact_path(turn, index, tool_name))
        self._archive_json(root, relative, payload)
        return relative.as_posix()

    def _archive_tool_completion(
        self,
        root: Path,
        turn: int,
        index: int,
        tool_name: str,
        call: dict[str, Any],
        arguments_sha256: str,
        payload: dict[str, Any],
        workspace_fingerprint: str,
    ) -> None:
        artifact = self._tool_artifact_path(turn, index, tool_name)
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise HarnessError("tool completion has no durable call id")
        if not re.fullmatch(r"[0-9a-f]{64}", arguments_sha256):
            raise HarnessError("tool completion has an invalid argument digest")
        if not re.fullmatch(r"[0-9a-f]{64}", workspace_fingerprint):
            raise HarnessError("tool completion has an invalid workspace fingerprint")
        encoded = _session_read_bytes(root, Path(artifact))
        receipt = {
            "schema_version": TOOL_COMPLETION_SCHEMA_VERSION,
            "turn": turn,
            "index": index,
            "tool": tool_name,
            "call_id": call_id,
            "arguments_sha256": arguments_sha256,
            "artifact": artifact,
            "result_sha256": sha256_bytes(encoded),
            "workspace_fingerprint": workspace_fingerprint,
            "completed": True,
        }
        relative = Path(self._tool_receipt_path(turn, index, tool_name))
        self._archive_json(root, relative, receipt)

    def _load_tool_completion(
        self,
        root: Path,
        turn: int,
        index: int,
        tool_name: str,
        call: dict[str, Any],
        arguments_sha256: str,
        tools: WorkspaceTools,
    ) -> dict[str, Any] | None:
        receipt_relative = Path(self._tool_receipt_path(turn, index, tool_name))
        try:
            encoded_receipt = _session_read_bytes(root, receipt_relative)
        except FileNotFoundError:
            return None
        try:
            receipt = json.loads(encoded_receipt)
        except (OSError, json.JSONDecodeError) as error:
            raise HarnessError("tool completion receipt is malformed") from error
        required_fields = {
            "schema_version",
            "turn",
            "index",
            "tool",
            "call_id",
            "arguments_sha256",
            "artifact",
            "result_sha256",
            "workspace_fingerprint",
            "completed",
        }
        if not isinstance(receipt, dict) or set(receipt) != required_fields:
            raise HarnessError("tool completion receipt schema is invalid")
        expected = {
            "schema_version": TOOL_COMPLETION_SCHEMA_VERSION,
            "turn": turn,
            "index": index,
            "tool": tool_name,
            "call_id": call.get("id"),
            "arguments_sha256": arguments_sha256,
            "artifact": self._tool_artifact_path(turn, index, tool_name),
            "completed": True,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise HarnessError("tool completion receipt identity mismatch")
        fingerprint = receipt.get("workspace_fingerprint")
        result_digest = receipt.get("result_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise HarnessError("tool completion workspace fingerprint is missing or invalid")
        if not isinstance(result_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", result_digest):
            raise HarnessError("tool completion result digest is missing or invalid")
        try:
            encoded = _session_read_bytes(root, Path(expected["artifact"]))
        except OSError as error:
            raise HarnessError("tool completion artifact is missing or unsafe") from error
        if sha256_bytes(encoded) != result_digest:
            raise HarnessError("tool completion artifact hash mismatch")
        if tools.workspace_fingerprint() != fingerprint:
            return None
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise HarnessError("tool completion artifact is malformed") from error
        if not isinstance(payload, dict):
            raise HarnessError("tool completion artifact is not an object")
        return payload

    @staticmethod
    def _tool_artifact_path(turn: int, index: int, tool_name: str) -> str:
        safe_tool = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_name)[:80]
        return (Path("tool-results") / f"turn-{turn:04d}-{index:03d}-{safe_tool}.json").as_posix()

    @staticmethod
    def _tool_receipt_path(turn: int, index: int, tool_name: str) -> str:
        safe_tool = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_name)[:80]
        return (Path("tool-receipts") / f"turn-{turn:04d}-{index:03d}-{safe_tool}.json").as_posix()

    @staticmethod
    def _context_tool_payload(payload: dict[str, Any], artifact: str) -> dict[str, Any]:
        encoded = canonical_json(payload).encode("utf-8")
        digest = sha256_bytes(encoded)
        if len(encoded) <= MAX_CONTEXT_TOOL_BYTES:
            value = dict(payload)
            value["_artifact"] = artifact
            value["_sha256"] = digest
            return value
        preview, _ = _bounded_text(encoded, MAX_CONTEXT_TOOL_BYTES)
        return {
            "ok": payload.get("ok") is True,
            "artifact": artifact,
            "sha256": digest,
            "bytes": len(encoded),
            "preview": preview,
            "truncated_for_context": True,
        }
