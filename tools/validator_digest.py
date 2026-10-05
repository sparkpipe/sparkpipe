#!/usr/bin/env python3
"""Digest of a GPU validator source and the validation templates it includes.

The validators include templates from include/sparkpipe/family/validation/.
A receipt that pinned only the .cu digest would not change when a template
changes what validates a pack, so every validator pin hashes the .cu together
with every template it reaches through #include "sparkpipe/family/validation/...".

The digest is SHA-256 over one line per file, "<name>\t<sha256>\n": the
validator first under its basename, then each reached template under its
include path, sorted.
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_ROOT = ROOT / "include"
TEMPLATE_PREFIX = "sparkpipe/family/validation/"
INCLUDE_PATTERN = re.compile(r'^\s*#\s*include\s*"(' + re.escape(TEMPLATE_PREFIX) + r'[^"]+)"', re.M)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reached_templates(source: Path) -> list[str]:
    seen: set[str] = set()
    pending = [source]
    while pending:
        text = pending.pop().read_text(encoding="utf-8", errors="replace")
        for name in INCLUDE_PATTERN.findall(text):
            if name in seen:
                continue
            path = INCLUDE_ROOT / name
            if not path.is_file():
                raise SystemExit(f"{source}: includes {name}, which is not under {INCLUDE_ROOT}")
            seen.add(name)
            pending.append(path)
    return sorted(seen)


def validator_digest(source: Path) -> str:
    lines = [f"{source.name}\t{file_sha256(source)}\n"]
    lines += [f"{name}\t{file_sha256(INCLUDE_ROOT / name)}\n" for name in reached_templates(source)]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validator_digest.py VALIDATOR.cu", file=sys.stderr)
        return 2
    source = Path(sys.argv[1])
    if not source.is_file():
        print(f"validator_digest: {source} is not a file", file=sys.stderr)
        return 2
    print(validator_digest(source.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
