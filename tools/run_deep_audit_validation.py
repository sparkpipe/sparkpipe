#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as datetime_module
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO


FINGERPRINT_EXCLUDED_DIRECTORY_NAMES = {
    ".audit",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "build",
}
FINGERPRINT_EXCLUDED_PATHS = {
    "PACKAGE_MANIFEST.json",
    "SHA256SUMS",
    "docs/PROPOSED_CHANGE_MANIFEST.md",
    "docs/VALIDATION_STATUS.json",
    "docs/VALIDATION_STATUS.md",
}
FINGERPRINT_EXCLUDED_PREFIXES = (
    "docs/validation-logs/",
)


@dataclass(frozen=True)
class ValidationStep:
    name: str
    category: str
    command: tuple[str, ...]
    timeout_seconds: int
    optional_tool: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    name: str
    category: str
    status: str
    return_code: int | None
    duration_seconds: float
    command: list[str]
    log_path: str
    note: str


class ValidationRunLock:
    def __init__(self, repository_root: Path) -> None:
        self.lock_path = repository_root / ".audit" / "deep-validation.lock"
        self.lock_file: TextIO | None = None

    def __enter__(self) -> "ValidationRunLock":
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
                f"another deep validation run owns {self.lock_path}: "
                f"{owner_description}"
            ) from error

        self.lock_file.seek(0)
        self.lock_file.truncate()
        self.lock_file.write(f"pid={os.getpid()}\n")
        self.lock_file.flush()
        os.fsync(self.lock_file.fileno())
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        if self.lock_file is None:
            return
        fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
        self.lock_file.close()
        self.lock_file = None


def path_is_fingerprinted(relative_path: Path) -> bool:
    relative_text = relative_path.as_posix()
    if any(
        part in FINGERPRINT_EXCLUDED_DIRECTORY_NAMES
        for part in relative_path.parts
    ):
        return False
    if relative_text in FINGERPRINT_EXCLUDED_PATHS:
        return False
    if any(
        relative_text.startswith(prefix)
        for prefix in FINGERPRINT_EXCLUDED_PREFIXES
    ):
        return False
    return True


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_fingerprint(repository_root: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(repository_root.rglob("*")):
        relative_path = path.relative_to(repository_root)
        if not path_is_fingerprinted(relative_path):
            continue
        relative_bytes = relative_path.as_posix().encode("utf-8")
        if path.is_symlink():
            entry_type = b"symlink"
            identity = os.readlink(path).encode("utf-8")
        elif path.is_file():
            entry_type = b"file"
            identity = hash_file(path).encode("ascii")
            file_count += 1
        elif path.is_dir():
            entry_type = b"directory"
            identity = b""
        else:
            raise RuntimeError(f"unsupported repository entry: {path}")
        digest.update(relative_bytes)
        digest.update(b"\0")
        digest.update(entry_type)
        digest.update(b"\0")
        digest.update(identity)
        digest.update(b"\n")
    return digest.hexdigest(), file_count


def build_source_stability_result(
    repository_root: Path,
    log_directory: Path,
    input_fingerprint: str,
    input_file_count: int,
) -> tuple[ValidationResult, str, int]:
    start_time = time.monotonic()
    output_fingerprint, output_file_count = source_tree_fingerprint(
        repository_root
    )
    stable = (
        output_fingerprint == input_fingerprint
        and output_file_count == input_file_count
    )
    status = "passed" if stable else "failed"
    note = (
        f"input={input_fingerprint} output={output_fingerprint} "
        f"input_files={input_file_count} output_files={output_file_count}"
    )
    log_path = log_directory / "source_tree_stability.log"
    log_path.write_text(note + "\n", encoding="utf-8")
    result = ValidationResult(
        name="source_tree_stability",
        category="preflight",
        status=status,
        return_code=0 if stable else 1,
        duration_seconds=round(time.monotonic() - start_time, 3),
        command=["internal", "source-tree-fingerprint"],
        log_path=log_path.relative_to(repository_root).as_posix(),
        note=note,
    )
    return result, output_fingerprint, output_file_count


def build_validation_steps(jobs: int) -> list[ValidationStep]:
    parallel = f"-j{jobs}"
    return [
        ValidationStep(
            name="clean_build",
            category="preflight",
            command=("make", "clean"),
            timeout_seconds=300,
        ),
        ValidationStep(
            name="python_tool_syntax",
            category="tooling",
            command=(
                sys.executable,
                "-m",
                "py_compile",
                "tools/audit_core_boundaries.py",
                "tools/generate_proposed_change_manifest.py",
                "tools/package_audited_proposal.py",
                "tools/run_deep_audit_validation.py",
                "tests/test_model_driver_contracts.py",
            ),
            timeout_seconds=120,
        ),
        ValidationStep(
            name="host_build",
            category="build",
            command=("make", parallel, "all"),
            timeout_seconds=600,
        ),
        ValidationStep(
            name="core_boundary_audit",
            category="architecture",
            command=("make", "architecture_audit"),
            timeout_seconds=180,
        ),
        ValidationStep(
            name="non_glm_model_driver_contracts",
            category="driver-contract",
            command=("make", "model_driver_contracts"),
            timeout_seconds=300,
        ),
        ValidationStep(
            name="memory_contracts",
            category="contract",
            command=(sys.executable, "tests/test_memory_contracts.py"),
            timeout_seconds=180,
        ),
        ValidationStep(
            name="host_test_suite",
            category="test",
            command=("make", parallel, "test"),
            timeout_seconds=1200,
        ),
        ValidationStep(
            name="required_host_targets",
            category="build",
            command=(
                "make",
                parallel,
                "glm52_pp13_service_backend",
                "tools",
            ),
            timeout_seconds=600,
        ),
        ValidationStep(
            name="cuda_node_context_builder",
            category="optional-cuda-build",
            command=("make", parallel, "glm52_pp13_node_context_builder"),
            timeout_seconds=600,
            optional_tool="nvcc",
        ),
    ]


def build_skipped_result(
    repository_root: Path,
    log_directory: Path,
    step: ValidationStep,
    note: str,
) -> ValidationResult:
    log_path = log_directory / f"{step.name}.log"
    log_path.write_text(note + "\n", encoding="utf-8")
    return ValidationResult(
        name=step.name,
        category=step.category,
        status="skipped",
        return_code=None,
        duration_seconds=0.0,
        command=list(step.command),
        log_path=log_path.relative_to(repository_root).as_posix(),
        note=note,
    )


def run_step(
    repository_root: Path,
    log_directory: Path,
    step: ValidationStep,
) -> ValidationResult:
    log_path = log_directory / f"{step.name}.log"
    step_environment = os.environ.copy()
    step_environment["PYTHONPYCACHEPREFIX"] = str(
        repository_root / "build" / "python-cache"
    )
    if step.optional_tool is not None and shutil.which(step.optional_tool) is None:
        return build_skipped_result(
            repository_root,
            log_directory,
            step,
            f"{step.optional_tool} is unavailable; this is not a pass",
        )

    start_time = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        try:
            completed = subprocess.run(
                step.command,
                cwd=repository_root,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=step.timeout_seconds,
                env=step_environment,
            )
            duration_seconds = time.monotonic() - start_time
            status = "passed" if completed.returncode == 0 else "failed"
            note = ""
            return_code = completed.returncode
        except subprocess.TimeoutExpired:
            duration_seconds = time.monotonic() - start_time
            log_file.write(
                f"\nTIMEOUT after {step.timeout_seconds} seconds\n"
            )
            status = "failed"
            note = f"timed out after {step.timeout_seconds} seconds"
            return_code = None

    return ValidationResult(
        name=step.name,
        category=step.category,
        status=status,
        return_code=return_code,
        duration_seconds=round(duration_seconds, 3),
        command=list(step.command),
        log_path=log_path.relative_to(repository_root).as_posix(),
        note=note,
    )


def run_validation_steps(
    repository_root: Path,
    log_directory: Path,
    steps: list[ValidationStep],
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    clean_failed = False

    for step in steps:
        if clean_failed:
            results.append(
                build_skipped_result(
                    repository_root,
                    log_directory,
                    step,
                    "skipped because the clean-build preflight failed",
                )
            )
            continue

        result = run_step(repository_root, log_directory, step)
        results.append(result)
        if step.name == "clean_build" and result.status == "failed":
            clean_failed = True

    return results


def write_markdown(
    output_path: Path,
    generated_at: str,
    results: list[ValidationResult],
    input_fingerprint: str,
    output_fingerprint: str,
    input_file_count: int,
    output_file_count: int,
) -> None:
    lines = [
        "# Validation Status",
        "",
        f"Generated at `{generated_at}`.",
        "",
        "This is a host-side receipt for the audited source tree. It does not "
        "qualify CUDA execution, numerical correctness, latency, throughput, "
        "or production readiness.",
        "",
        "The harness acquires an exclusive repository lock, fingerprints the "
        "source tree, and begins with `make clean`, so a passing receipt cannot "
        "be inherited from an incremental or concurrent build. The final "
        "fingerprint must match before the receipt is issued.",
        "",
        f"Input source fingerprint: `{input_fingerprint}` ({input_file_count} files).",
        "",
        f"Output source fingerprint: `{output_fingerprint}` ({output_file_count} files).",
        "",
        "| Step | Category | Status | Return code | Duration | Log |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        return_code = "—" if result.return_code is None else str(result.return_code)
        lines.append(
            f"| `{result.name}` | {result.category} | **{result.status}** | "
            f"{return_code} | {result.duration_seconds:.3f}s | "
            f"`{result.log_path}` |"
        )
    lines.extend(("", "## Commands", ""))
    for result in results:
        lines.extend((
            f"### `{result.name}`",
            "",
            "```sh",
            " ".join(result.command),
            "```",
            "",
        ))
        if result.note:
            lines.extend((f"Note: {result.note}", ""))
    lines.extend((
        "## Interpretation",
        "",
        "- `passed` means only that the named host command returned success.",
        "- `failed` is a release blocker for this proposal.",
        "- `skipped` is not a pass; the required local tool was unavailable or a preflight blocked the step.",
        "- Non-GLM drivers remain `NOT_MEASURED` until exact-hardware GPU receipts exist.",
        "",
    ))
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_validation_receipts(
    repository_root: Path,
    results: list[ValidationResult],
    input_fingerprint: str,
    output_fingerprint: str,
    input_file_count: int,
    output_file_count: int,
) -> None:
    generated_at = datetime_module.datetime.now(
        datetime_module.timezone.utc
    ).isoformat()
    document = {
        "schema_version": 3,
        "generated_at": generated_at,
        "repository_root": str(repository_root),
        "host_validation_only": True,
        "clean_build_required": True,
        "exclusive_run_lock": True,
        "source_tree_stability_required": True,
        "input_source_fingerprint": input_fingerprint,
        "output_source_fingerprint": output_fingerprint,
        "input_source_file_count": input_file_count,
        "output_source_file_count": output_file_count,
        "fingerprint_excluded_paths": sorted(FINGERPRINT_EXCLUDED_PATHS),
        "fingerprint_excluded_prefixes": list(FINGERPRINT_EXCLUDED_PREFIXES),
        "results": [asdict(result) for result in results],
    }
    output_directory = repository_root / "docs"
    (output_directory / "VALIDATION_STATUS.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(
        output_directory / "VALIDATION_STATUS.md",
        generated_at,
        results,
        input_fingerprint,
        output_fingerprint,
        input_file_count,
        output_file_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the SparkPipe core-boundary and model-driver host audit."
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    arguments = parser.parse_args()

    repository_root = arguments.repository_root.resolve()
    if not (repository_root / "Makefile").is_file():
        raise RuntimeError(f"not a SparkPipe repository: {repository_root}")
    if arguments.jobs < 1:
        raise RuntimeError("--jobs must be at least 1")

    with ValidationRunLock(repository_root):
        output_directory = repository_root / "docs"
        log_directory = output_directory / "validation-logs"
        shutil.rmtree(log_directory, ignore_errors=True)
        log_directory.mkdir(parents=True, exist_ok=True)

        input_fingerprint, input_file_count = source_tree_fingerprint(
            repository_root
        )
        results = run_validation_steps(
            repository_root,
            log_directory,
            build_validation_steps(arguments.jobs),
        )
        (
            source_stability_result,
            output_fingerprint,
            output_file_count,
        ) = build_source_stability_result(
            repository_root,
            log_directory,
            input_fingerprint,
            input_file_count,
        )
        results.append(source_stability_result)
        write_validation_receipts(
            repository_root,
            results,
            input_fingerprint,
            output_fingerprint,
            input_file_count,
            output_file_count,
        )

    failed_results = [result for result in results if result.status == "failed"]
    for result in results:
        print(
            f"{result.status.upper():7s} {result.name} "
            f"({result.duration_seconds:.3f}s)"
        )
    return 1 if failed_results else 0


if __name__ == "__main__":
    raise SystemExit(main())
