#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "generate_package_manifest.py"
CHECKSUM_GENERATOR = ROOT / "tools" / "generate_sha256sums.py"
VERIFIER = ROOT / "tools" / "verify_package_manifest.py"
PACKAGE_BUILDER = ROOT / "tools" / "package_source.py"
ARCHIVE_VERIFIER = ROOT / "tools" / "verify_source_archive.py"
PACKAGE_INVENTORY = ROOT / "tools" / "package_inventory.py"


def run(*arguments: str, expect_success: bool) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if (completed.returncode == 0) != expect_success:
        raise AssertionError(
            f"unexpected status {completed.returncode} for {arguments}:\n"
            f"{completed.stdout}"
        )
    return completed


def verify_manifest_rejects_payload_drift() -> None:
    with tempfile.TemporaryDirectory(prefix="sparkpipe-package-test-") as temporary:
        package_root = Path(temporary)
        (package_root / "cache").mkdir()
        (package_root / "cache" / "kv_cache.c").write_text("int cache_value;\n")
        (package_root / "source.c").write_text("int source_value;\n")
        (package_root / "build").mkdir()
        (package_root / "build" / "ignored.o").write_bytes(b"build")
        (package_root / "qualification").mkdir()
        (package_root / "qualification" / "evidence.log").write_text("evidence\n")
        (package_root / "qualification" / "ds4_eval" / "runs").mkdir(parents=True)
        (package_root / "qualification" / "ds4_eval" / "runs" / "summary.json").write_text(
            "{\"score\": 1}\n"
        )
        (package_root / "qualification" / "glm52.latest.txt").write_text(
            "generated receipt\n"
        )
        for private_directory in (".agents",".codex","nested/.tshome"):
            path = package_root / private_directory
            path.mkdir(parents=True)
            (path / "private-state").write_text("must not ship\n")

        run(
            "python3",
            str(GENERATOR),
            "--root",
            str(package_root),
            expect_success=True,
        )
        run(
            "python3",
            str(CHECKSUM_GENERATOR),
            "--root",
            str(package_root),
            expect_success=True,
        )
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            expect_success=True,
        )
        manifest = json.loads((package_root / "PACKAGE_MANIFEST.json").read_text())
        packaged_paths = {entry["path"] for entry in manifest["files"]}
        if any("private-state" in path for path in packaged_paths):
            raise AssertionError("private controller state entered the source manifest")

        (package_root / "unlisted.c").write_text("int unlisted;\n")
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            expect_success=False,
        )
        (package_root / "unlisted.c").unlink()

        (package_root / "source.c").write_text("int source_value_changed;\n")
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            expect_success=False,
        )
        (package_root / "source.c").write_text("int source_value;\n")

        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            "--strict-package",
            expect_success=False,
        )
        (package_root / "build" / "ignored.o").unlink()
        (package_root / "build").rmdir()
        (package_root / "qualification" / "evidence.log").unlink()
        shutil.rmtree(package_root / "qualification")
        shutil.rmtree(package_root / ".agents")
        shutil.rmtree(package_root / ".codex")
        shutil.rmtree(package_root / "nested")
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            "--strict-package",
            expect_success=True,
        )

        (package_root / "nested.tar.gz").write_bytes(b"not an archive")
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            "--strict-package",
            expect_success=False,
        )
        (package_root / "nested.tar.gz").unlink()

        symlink_path = package_root / "source-link.c"
        symlink_path.symlink_to(package_root / "source.c")
        run(
            "python3",
            str(VERIFIER),
            "--root",
            str(package_root),
            "--strict-package",
            expect_success=False,
        )
        symlink_path.unlink()


def verify_archive_round_trip_does_not_contaminate_payload() -> None:
    with tempfile.TemporaryDirectory(prefix="sparkpipe-archive-test-") as temporary:
        temporary_root = Path(temporary)
        package_root = temporary_root / "package"
        tools_root = package_root / "tools"
        extraction_root = temporary_root / "extracted"
        archive_path = temporary_root / "source.tar.gz"

        tools_root.mkdir(parents=True)
        (package_root / "source.c").write_text("int source_value;\n")
        shutil.copy2(PACKAGE_INVENTORY, tools_root / PACKAGE_INVENTORY.name)
        shutil.copy2(VERIFIER, tools_root / VERIFIER.name)

        run(
            "python3",
            str(GENERATOR),
            "--root",
            str(package_root),
            expect_success=True,
        )
        run(
            "python3",
            str(CHECKSUM_GENERATOR),
            "--root",
            str(package_root),
            expect_success=True,
        )
        run(
            "python3",
            str(PACKAGE_BUILDER),
            "--root",
            str(package_root),
            "--output",
            str(archive_path),
            "--prefix",
            "test-package",
            expect_success=True,
        )
        run(
            "python3",
            str(ARCHIVE_VERIFIER),
            str(archive_path),
            "--extract-to",
            str(extraction_root),
            expect_success=True,
        )

        extracted_package = extraction_root / "test-package"
        contaminants = [
            path.relative_to(extracted_package).as_posix()
            for path in extracted_package.rglob("*")
            if path.name == "__pycache__" or path.suffix == ".pyc"
        ]
        if contaminants:
            raise AssertionError(
                "archive verification contaminated the extracted payload: "
                + ", ".join(contaminants)
            )


def main() -> int:
    verify_manifest_rejects_payload_drift()
    verify_archive_round_trip_does_not_contaminate_payload()
    print("package manifest generation and Git-independent verification pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
