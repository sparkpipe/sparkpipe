#!/usr/bin/env python3

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = (
    (
        ROOT
        / "modules/glm5_next_resident_decode_stage/source"
        / "spark_glm5_next_resident_decode_stage_module.c",
        (),
    ),
    (
        ROOT
        / "modules/glm5_next_resident_decode_stage/source"
        / "spark_glm5_next_serving_adapter.c",
        (),
    ),
    (
        ROOT
        / "modules/glm5_next_resident_decode_stage/validation"
        / "spark_glm5_next_resident_decode_stage_mtp_parity.cu",
        ("-x", "c++", "-std=c++17"),
    ),
)


def main() -> int:
    compiler = shlex.split(os.environ.get("CC", "cc"))
    for source, overrides in SOURCES:
        command = (
            compiler
            + [
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-D_GNU_SOURCE",
                "-D_POSIX_C_SOURCE=200809L",
                "-D_FILE_OFFSET_BITS=64",
                "-DSPARK_BATCH_BUCKET=1024u",
                "-fsyntax-only",
                "-I.",
                "-Iinclude",
                "-Isrc",
                "-Itests/cuda_stub",
                "-Imodel-families/common/include",
                "-Imodel-families/glm5_next/include",
                "-Imodules/glm5_next_resident_decode_stage/include",
                "-Imodules/glm5_next_resident_decode_stage/source",
                "-include",
                "model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h",
                "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5",
                '-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"',
                "-DGLM5_NEXT_MODEL_REVISION=\"test-revision\"",
                "-DGLM5_NEXT_CONTRACT_SHA256="
                "\"0000000000000000000000000000000000000000000000000000000000000000\"",
            ]
            + list(overrides)
            + [str(source)]
        )
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
            return result.returncode
    print("PASS GLM5_NEXT resident module host syntax")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
