#!/usr/bin/env python3

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "modules/dsv4_resident_decode_stage/source"
    / "spark_dsv4_resident_decode_stage_module.c"
)


def main() -> int:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    command = compiler + [
        "-std=c11",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fsyntax-only",
        "-Iinclude",
        "-Isrc",
        "-Itests/cuda_stub",
        "-Imodel-families/dsv4/include",
        "-Imodules/dsv4_resident_decode_stage/include",
        "-Imodules/dsv4_resident_decode_stage/source",
        "-include",
        "model-families/dsv4/include/sparkpipe/spark_dsv4_model.h",
        str(SOURCE),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return(result.returncode)
    print("PASS DSV4 resident module host syntax")
    return(0)


if __name__ == "__main__":
    raise SystemExit(main())
