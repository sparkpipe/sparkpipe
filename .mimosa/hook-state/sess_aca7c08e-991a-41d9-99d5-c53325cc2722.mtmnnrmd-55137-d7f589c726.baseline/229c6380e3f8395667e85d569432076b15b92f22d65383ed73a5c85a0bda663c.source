#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

MANIFEST_NAME = "RUNTIME_COMPLETION_OVERLAY.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while True:
            block = input_file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def safe_relative_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"unsafe overlay path: {path_text!r}")
    return path


def copy_file_atomic(source: Path, destination: Path, mode: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        shutil.copyfile(source, temporary_path)
        os.chmod(temporary_path, mode)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the runtime-completion affected-files overlay."
    )
    parser.add_argument("overlay_root", type=Path)
    parser.add_argument("target_root", type=Path)
    arguments = parser.parse_args()
    overlay_root = arguments.overlay_root.resolve()
    target_root = arguments.target_root.resolve()
    manifest_path = overlay_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise SystemExit(f"missing overlay manifest: {manifest_path}")
    if not target_root.is_dir():
        raise SystemExit(f"target root is not a directory: {target_root}")
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("schema_version") != 1:
        raise SystemExit("unsupported overlay manifest schema")
    for record in manifest.get("base_files", []):
        relative_path = safe_relative_path(record["path"])
        target_path = target_root / relative_path
        if not target_path.is_file():
            raise SystemExit(f"missing Phase 10 base file: {relative_path}")
        if target_path.stat().st_size != record["bytes"]:
            raise SystemExit(f"Phase 10 base byte mismatch: {relative_path}")
        if sha256_file(target_path) != record["sha256"]:
            raise SystemExit(f"Phase 10 base SHA-256 mismatch: {relative_path}")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise SystemExit("overlay manifest has no files")
    for record in records:
        relative_path = safe_relative_path(record["path"])
        source_path = overlay_root / relative_path
        destination_path = target_root / relative_path
        if not source_path.is_file():
            raise SystemExit(f"missing overlay payload: {relative_path}")
        if source_path.stat().st_size != record["bytes"]:
            raise SystemExit(f"overlay byte mismatch: {relative_path}")
        if sha256_file(source_path) != record["sha256"]:
            raise SystemExit(f"overlay SHA-256 mismatch: {relative_path}")
        copy_file_atomic(source_path, destination_path, record["mode"])
    for record in records:
        relative_path = safe_relative_path(record["path"])
        destination_path = target_root / relative_path
        if destination_path.stat().st_size != record["bytes"]:
            raise SystemExit(f"installed byte mismatch: {relative_path}")
        if sha256_file(destination_path) != record["sha256"]:
            raise SystemExit(f"installed SHA-256 mismatch: {relative_path}")
    print(f"PASS applied {len(records)} runtime-completion files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
