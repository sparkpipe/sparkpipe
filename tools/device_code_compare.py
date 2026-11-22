#!/usr/bin/env python3
"""Compare the device code of two built trees, cubin by cubin.

A change that must leave every kernel as it was (a call-site rename, a launch
path change) is checked by building the same targets in a tree at the base
revision and in a tree at the head revision, then running this tool. It
extracts every cubin embedded in each artifact with `cuobjdump -xelf all`,
plus any loose `.cubin` files under the given directories, and compares them
byte for byte. Any difference, or a cubin present on one side only, fails.

Usage:
    python3 tools/device_code_compare.py BASE_DIR HEAD_DIR [--cuobjdump PATH]

Each directory is searched recursively for shared libraries, executables and
`.cubin` files; artifacts are paired by their path relative to the directory.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ELF_MAGIC = b"\x7fELF"


def candidates(root):
    found = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".cubin":
            found[relative] = ("cubin", path)
            continue
        if path.suffix in (".so", ".o", ".a") or os.access(path, os.X_OK):
            with path.open("rb") as handle:
                if handle.read(4) == ELF_MAGIC or path.suffix == ".a":
                    found[relative] = ("artifact", path)
    return found


def extract(cuobjdump, path, scratch):
    directory = Path(tempfile.mkdtemp(dir=scratch))
    result = subprocess.run([cuobjdump, "-xelf", "all", str(path.resolve())], cwd=directory, capture_output=True, text=True)
    if result.returncode != 0 and "does not contain device code" not in result.stderr:
        raise RuntimeError("%s: cuobjdump failed: %s" % (path, result.stderr.strip()))
    cubins = {}
    for cubin in sorted(directory.iterdir()):
        cubins[cubin.name] = hashlib.sha256(cubin.read_bytes()).hexdigest()
    return cubins


def digests(cuobjdump, root, scratch):
    table = {}
    for relative, (kind, path) in candidates(root).items():
        if kind == "cubin":
            table[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            continue
        for name, digest in extract(cuobjdump, path, scratch).items():
            table[relative + ":" + name] = digest
    return table


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path)
    parser.add_argument("head", type=Path)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump") or "/usr/local/cuda/bin/cuobjdump")
    options = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="device-code-") as scratch:
        base = digests(options.cuobjdump, options.base, scratch)
        head = digests(options.cuobjdump, options.head, scratch)
    failures = 0
    for name in sorted(set(base) | set(head)):
        if name not in head:
            print("ONLY-BASE %s" % name)
        elif name not in base:
            print("ONLY-HEAD %s" % name)
        elif base[name] != head[name]:
            print("DIFFER %s" % name)
        else:
            continue
        failures += 1
    print("cubins base %d head %d identical %d different %d" % (len(base), len(head), sum(1 for name in base if head.get(name) == base[name]), failures))
    if not base:
        print("no device code found under %s" % options.base)
        return 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
