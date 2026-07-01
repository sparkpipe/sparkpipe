#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import Iterable, Sequence


SPARK_HOME_RE = re.compile(r"/home/spark[0-9a-z]+/")


def replace_spark_home(text: str, spark_user: str) -> str:
    return SPARK_HOME_RE.sub(f"/home/{spark_user}/", text)


def rewrite_file(path: Path, spark_user: str) -> bool:
    original = path.read_text(encoding="utf-8")
    rewritten = replace_spark_home(original, spark_user)
    if rewritten == original:
        return False
    path.write_text(rewritten, encoding="utf-8")
    return True


def target_files(generated_dir: Path) -> Iterable[Path]:
    yield generated_dir / "tvm_ffi_flags.mk"
    yield generated_dir / "runtime_link_args.txt"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Relocate GLM52 B12x generated AOT metadata to this Spark user")
    parser.add_argument("--generated-dir", required=True, type=Path)
    parser.add_argument("--spark-user", required=True)
    args = parser.parse_args(argv)
    generated_dir = args.generated_dir.resolve()
    if not generated_dir.is_dir():
        print(f"missing generated dir: {generated_dir}", file=sys.stderr)
        return 2
    changed = []
    for path in target_files(generated_dir):
        if not path.is_file():
            print(f"missing AOT metadata file: {path}", file=sys.stderr)
            return 3
        if rewrite_file(path, args.spark_user):
            changed.append(str(path))
    print(f"relocated_files={len(changed)}")
    for path in changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
