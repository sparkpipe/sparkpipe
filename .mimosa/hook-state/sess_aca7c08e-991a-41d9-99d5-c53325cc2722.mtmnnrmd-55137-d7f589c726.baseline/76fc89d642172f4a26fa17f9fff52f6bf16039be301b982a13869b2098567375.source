#!/usr/bin/env python3
"""Build a deterministic, single-root SparkPipe source archive."""

from __future__ import annotations

import argparse
import gzip
import io
import tarfile
from pathlib import Path, PurePosixPath

from package_inventory import (
    CHECKSUM_MANIFEST_NAME,
    PACKAGE_MANIFEST_NAME,
    source_inventory,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREFIX = "sparkpipe-phase7-foundational-safety"
FIXED_MTIME = 0


def normalized_mode(path: Path) -> int:
    return 0o755 if (path.stat().st_mode & 0o111) != 0 else 0o644


def directory_names(relative_paths: list[str]) -> list[str]:
    names: set[str] = set()
    for relative_path in relative_paths:
        parent = PurePosixPath(relative_path).parent
        while str(parent) != ".":
            names.add(parent.as_posix())
            parent = parent.parent
    return sorted(names, key=lambda value: (value.count("/"), value))


def tar_info(name: str, mode: int, size: int = 0, is_directory: bool = False) -> tarfile.TarInfo:
    information = tarfile.TarInfo(name=name)
    information.uid = 0
    information.gid = 0
    information.uname = "root"
    information.gname = "root"
    information.mtime = FIXED_MTIME
    information.mode = mode
    information.size = size
    information.type = tarfile.DIRTYPE if is_directory else tarfile.REGTYPE
    return information


def build_archive(root: Path, output_path: Path, prefix: str) -> None:
    inventory = source_inventory(root)
    relative_paths = [entry.path for entry in inventory]
    relative_paths.extend([PACKAGE_MANIFEST_NAME, CHECKSUM_MANIFEST_NAME])
    relative_paths = sorted(set(relative_paths))

    for metadata_name in (PACKAGE_MANIFEST_NAME, CHECKSUM_MANIFEST_NAME):
        if not (root / metadata_name).is_file():
            raise ValueError(f"missing package metadata: {metadata_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                archive.addfile(tar_info(prefix, 0o755, is_directory=True))
                for directory_name in directory_names(relative_paths):
                    archive.addfile(
                        tar_info(
                            f"{prefix}/{directory_name}",
                            0o755,
                            is_directory=True,
                        )
                    )
                for relative_path in relative_paths:
                    source_path = root / relative_path
                    data = source_path.read_bytes()
                    information = tar_info(
                        f"{prefix}/{relative_path}",
                        normalized_mode(source_path),
                        size=len(data),
                    )
                    archive.addfile(information, io.BytesIO(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    arguments = parser.parse_args()

    prefix = arguments.prefix.strip("/")
    if not prefix or "/" in prefix or prefix in {".", ".."}:
        raise SystemExit("archive prefix must be one safe path component")
    build_archive(arguments.root.resolve(), arguments.output.resolve(), prefix)
    print(f"wrote {arguments.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
